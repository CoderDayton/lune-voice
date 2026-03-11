use std::sync::Arc;

use lune_app::{SessionManager, SessionManagerConfig};
use lune_asr::{AsrRegistry, OrtForcedAligner};
use lune_commands::VoiceCommandRouter;
use lune_core::asr::AsrSessionConfig;
use lune_core::config::{AsrBackendConfig, LlmBackendConfig, UserSettings};
use lune_core::llm::TextPostProcessor;
use lune_core::session::SessionEvent;
use lune_dict::DictionaryManager;
use lune_llm::{LlamaCppBackend, LlamaCppConfig};
use lune_persist::{AppPaths, Database, SettingsStore, ensure_default_models};
use tauri::{AppHandle, Emitter, ipc::Channel};
use tokio::sync::{Mutex, RwLock, mpsc};

use crate::dto::SessionStreamEvent;

pub struct AppRuntime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    app: AppHandle,
    pipeline: Mutex<LivePipeline>,
    status: RwLock<String>,
    session_stream: RwLock<Option<Channel<SessionStreamEvent>>>,
}

struct LivePipeline {
    paths: AppPaths,
    manager: Arc<SessionManager<AsrRegistry>>,
    active: bool,
    session_id: Option<String>,
    reload_required: bool,
}

impl AppRuntime {
    pub async fn new(app: AppHandle) -> Result<Self, String> {
        let paths = resolve_paths()?;
        let pipeline = build_pipeline(paths.clone()).await?;

        Ok(Self {
            inner: Arc::new(RuntimeInner {
                app,
                pipeline: Mutex::new(pipeline),
                status: RwLock::new("ready".into()),
                session_stream: RwLock::new(None),
            }),
        })
    }

    pub async fn service_status(&self) -> String {
        self.inner.status.read().await.clone()
    }

    pub async fn set_session_stream(
        &self,
        channel: Channel<SessionStreamEvent>,
    ) -> Result<(), String> {
        {
            let mut stream = self.inner.session_stream.write().await;
            *stream = Some(channel);
        }

        self.push_session_event(SessionStreamEvent::ServiceStatus {
            status: self.service_status().await,
        })
        .await;

        let active_session_id = {
            let pipeline = self.inner.pipeline.lock().await;
            pipeline.session_id.clone()
        };

        if let Some(session_id) = active_session_id {
            self.push_session_event(SessionStreamEvent::SessionStarted { session_id })
                .await;
        }

        Ok(())
    }

    pub async fn clear_session_stream(&self) {
        let mut stream = self.inner.session_stream.write().await;
        *stream = None;
    }

    pub async fn mark_reload_required(&self) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.reload_required = true;
    }

    pub async fn start_dictation(&self) -> Result<String, String> {
        let manager = self.prepare_manager_for_start().await?;

        let (event_tx, event_rx) = mpsc::channel(64);
        let session_id = match manager.start(event_tx).await {
            Ok(session_id) => session_id,
            Err(error) => {
                self.set_pipeline_session(None, false).await;
                self.publish_status("error").await;
                return Err(format!("start dictation: {error}"));
            }
        };

        let session_id = session_id.to_string();
        self.set_pipeline_session(Some(session_id.clone()), true)
            .await;

        self.push_session_event(SessionStreamEvent::SessionStarted {
            session_id: session_id.clone(),
        })
        .await;
        self.publish_status("listening").await;
        self.spawn_event_forwarder(event_rx);

        Ok(session_id)
    }

    pub async fn stop_dictation(&self) -> Result<(), String> {
        if let Some(manager) = self.active_manager().await {
            manager.stop().await;
        }
        Ok(())
    }

    pub async fn cancel_dictation(&self) -> Result<(), String> {
        if let Some(manager) = self.active_manager().await {
            manager.cancel().await;
        }
        Ok(())
    }

    pub async fn toggle_dictation(&self) -> Result<Option<String>, String> {
        let is_active = self.inner.pipeline.lock().await.active;

        if is_active {
            self.stop_dictation().await?;
            Ok(None)
        } else {
            self.start_dictation().await.map(Some)
        }
    }

    fn spawn_event_forwarder(&self, mut event_rx: mpsc::Receiver<SessionEvent>) {
        let app = self.inner.app.clone();
        let runtime = self.clone();

        tauri::async_runtime::spawn(async move {
            let mut saw_error = false;
            while let Some(event) = event_rx.recv().await {
                match event {
                    SessionEvent::Started => {
                        runtime.publish_status("listening").await;
                    }
                    SessionEvent::PartialText => {
                        runtime
                            .push_session_event(SessionStreamEvent::PartialText)
                            .await;
                    }
                    SessionEvent::Processing => {
                        runtime.publish_status("processing").await;
                        runtime
                            .push_session_event(SessionStreamEvent::Processing)
                            .await;
                    }
                    SessionEvent::FinalizedText(text) => {
                        runtime
                            .push_session_event(SessionStreamEvent::FinalizedText { text })
                            .await;
                    }
                    SessionEvent::CorrectedText(text) => {
                        runtime
                            .push_session_event(SessionStreamEvent::CorrectedText { text })
                            .await;
                    }
                    SessionEvent::CommandDetected(payload) => {
                        runtime
                            .push_session_event(SessionStreamEvent::CommandDetected { payload })
                            .await;
                    }
                    SessionEvent::Error(message) => {
                        saw_error = true;
                        runtime.finish_session().await;
                        runtime.publish_status("error").await;
                        runtime
                            .push_session_event(SessionStreamEvent::SessionError { message })
                            .await;
                    }
                    SessionEvent::Stopped => {
                        runtime.finish_session().await;
                        if !saw_error {
                            runtime.publish_status("ready").await;
                        }
                        runtime
                            .push_session_event(SessionStreamEvent::SessionStopped)
                            .await;
                        let _ = app.emit("lune://history-changed", ());
                    }
                    _ => {}
                }
            }
        });
    }

    async fn finish_session(&self) {
        self.set_pipeline_session(None, false).await;
    }

    async fn publish_status(&self, status: &str) {
        *self.inner.status.write().await = status.to_owned();
        self.push_session_event(SessionStreamEvent::ServiceStatus {
            status: status.to_owned(),
        })
        .await;
    }

    async fn push_session_event(&self, event: SessionStreamEvent) {
        let channel = self.inner.session_stream.read().await.clone();
        if let Some(channel) = channel {
            let _ = channel.send(event);
        }
    }

    async fn active_manager(&self) -> Option<Arc<SessionManager<AsrRegistry>>> {
        let pipeline = self.inner.pipeline.lock().await;
        pipeline.active.then(|| Arc::clone(&pipeline.manager))
    }

    async fn prepare_manager_for_start(&self) -> Result<Arc<SessionManager<AsrRegistry>>, String> {
        let reload_paths = {
            let pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Err("dictation session already active".into());
            }
            pipeline.reload_required.then(|| pipeline.paths.clone())
        };

        if let Some(paths) = reload_paths {
            let rebuilt = build_pipeline(paths).await?;
            let mut pipeline = self.inner.pipeline.lock().await;
            if pipeline.active {
                return Err("dictation session became active during reload".into());
            }
            *pipeline = rebuilt;
        }

        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.active = true;
        Ok(Arc::clone(&pipeline.manager))
    }

    async fn set_pipeline_session(&self, session_id: Option<String>, active: bool) {
        let mut pipeline = self.inner.pipeline.lock().await;
        pipeline.session_id = session_id;
        pipeline.active = active;
    }
}

impl Clone for AppRuntime {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

async fn build_pipeline(paths: AppPaths) -> Result<LivePipeline, String> {
    let mut settings = load_settings(&paths)?;
    patch_model_paths(&paths, &mut settings)?;
    init_ort_runtime(&settings);

    let asr = AsrRegistry::from_config(&settings.asr_backend, settings.device)
        .map_err(|e| format!("build asr backend: {e}"))?;
    let aligner = build_aligner(&asr, &settings)?;
    let post_processor = build_post_processor(&settings).await?;
    let router = Arc::new(VoiceCommandRouter::new());
    let dict = Arc::new(Mutex::new(
        DictionaryManager::new(
            Database::open(&paths.database_file()).map_err(|e| format!("open dict db: {e}"))?,
        )
        .map_err(|e| format!("init dictionary manager: {e}"))?,
    ));
    let db = Arc::new(Mutex::new(
        Database::open(&paths.database_file()).map_err(|e| format!("open session db: {e}"))?,
    ));
    let settings = Arc::new(RwLock::new(settings.clone()));

    let config = SessionManagerConfig {
        asr_backend_kind: asr.kind(),
        asr_config: AsrSessionConfig {
            language: Some(settings.read().await.language.clone()),
            ..Default::default()
        },
        pipeline: asr
            .pipeline_config()
            .with_overrides(&settings.read().await.pipeline),
    };

    let mut manager = SessionManager::new(
        Arc::new(asr),
        router,
        post_processor,
        dict,
        db,
        config,
        settings,
    )
    .with_vad_state_path(paths.adaptive_vad_file());

    if let Some(aligner) = aligner {
        manager = manager.with_aligner(aligner);
    }

    Ok(LivePipeline {
        paths,
        manager: Arc::new(manager),
        active: false,
        session_id: None,
        reload_required: false,
    })
}

fn resolve_paths() -> Result<AppPaths, String> {
    let paths = AppPaths::resolve().ok_or("failed to resolve app paths")?;
    paths
        .ensure_dirs()
        .map_err(|e| format!("ensure app dirs: {e}"))?;
    Ok(paths)
}

fn load_settings(paths: &AppPaths) -> Result<UserSettings, String> {
    SettingsStore::new(paths.settings_file())
        .load()
        .map_err(|e| format!("load settings: {e}"))
}

fn patch_model_paths(paths: &AppPaths, settings: &mut UserSettings) -> Result<(), String> {
    let resolved = ensure_default_models(
        &paths.models_dir(),
        &settings.asr_backend,
        &settings.llm_backend,
        settings.aligner.as_ref(),
    )
    .map_err(|e| format!("resolve models: {e}"))?;

    match &mut settings.asr_backend {
        AsrBackendConfig::ParakeetTdt { model_dir } if model_dir.as_os_str().is_empty() => {
            *model_dir = resolved.asr_model_dir;
        }
        AsrBackendConfig::OrtQwenAsr { model_dir } if model_dir.as_os_str().is_empty() => {
            *model_dir = resolved.ort_qwen_asr_model_dir;
        }
        _ => {}
    }

    match &mut settings.llm_backend {
        LlmBackendConfig::LlamaCppLocal { model_path, .. } if model_path.as_os_str().is_empty() => {
            *model_path = resolved.llm_model_path;
        }
        _ => {}
    }

    if let (Some(aligner), Some(model_dir)) = (&mut settings.aligner, resolved.aligner_model_dir)
        && aligner.model_dir.as_os_str().is_empty()
    {
        aligner.model_dir = model_dir;
    }

    Ok(())
}

fn init_ort_runtime(settings: &UserSettings) {
    let wants_gpu = settings.device.use_gpu(cfg!(feature = "cuda"));

    if wants_gpu {
        #[cfg(feature = "cuda")]
        {
            if ort::init()
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .commit()
            {
                return;
            }
        }
    }

    let _ = ort::init().commit();
}

fn build_aligner(
    asr: &AsrRegistry,
    settings: &UserSettings,
) -> Result<Option<Arc<OrtForcedAligner>>, String> {
    match (&settings.aligner, asr.as_ort_qwen()) {
        (Some(cfg), Some(backend)) if cfg.enabled => backend
            .make_forced_aligner(&cfg.model_dir)
            .map(|aligner| {
                let aligner = Arc::new(aligner);
                backend.attach_aligner(Arc::clone(&aligner));
                Some(aligner)
            })
            .map_err(|e| format!("build forced aligner: {e}")),
        _ => Ok(None),
    }
}

async fn build_post_processor(
    settings: &UserSettings,
) -> Result<Arc<dyn TextPostProcessor>, String> {
    match &settings.llm_backend {
        LlmBackendConfig::LlamaCppLocal {
            model_path,
            n_gpu_layers,
            n_ctx,
            max_output_tokens,
        } => LlamaCppBackend::new(LlamaCppConfig {
            model_path: model_path.clone(),
            n_gpu_layers: *n_gpu_layers,
            n_ctx: *n_ctx,
            n_threads: 4,
            max_output_tokens: *max_output_tokens,
        })
        .await
        .map(|backend| Arc::new(backend) as Arc<dyn TextPostProcessor>)
        .map_err(|e| format!("build llm backend: {e}")),
        _ => Err("OpenAI LLM is not supported by the desktop UI runtime yet".into()),
    }
}
