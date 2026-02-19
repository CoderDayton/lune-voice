mod session;

pub use session::{SessionManager, SessionManagerConfig};

use lune_persist::{AppPaths, ensure_default_models};
use tracing_subscriber::{EnvFilter, fmt};

// current_thread flavor + LocalSet: lets session tasks hold !Send types (cpal::Stream).
#[tokio::main(flavor = "current_thread")]
async fn main() {
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    tracing::info!("lune-voice starting");

    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        let paths = AppPaths::resolve().expect("resolve app paths");
        paths.ensure_dirs().expect("create app directories");

        let models = ensure_default_models(&paths.models_hub_dir())
            .expect("ensure models downloaded");

        tracing::info!(
            asr_dir = %models.asr_model_dir.display(),
            llm_path = %models.llm_model_path.display(),
            "models ready"
        );
    }).await;
}
