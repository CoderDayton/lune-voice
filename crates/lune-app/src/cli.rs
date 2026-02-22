use zbus::{Connection, Result, proxy};

#[proxy(
    interface = "com.lunevoice.Daemon1",
    default_service = "com.lunevoice.Daemon1",
    default_path = "/com/lunevoice/Daemon1"
)]
trait Daemon {
    async fn toggle(&self) -> Result<()>;
    async fn start(&self) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn cancel(&self) -> Result<()>;
    async fn quit(&self) -> Result<()>;
    async fn status(&self) -> Result<String>;
    async fn get_dictation_mode(&self) -> Result<String>;
    async fn set_dictation_mode(&self, mode: &str) -> Result<()>;
    async fn get_activation_mode(&self) -> Result<String>;
    async fn set_activation_mode(&self, mode: &str) -> Result<()>;
    async fn get_default_style(&self) -> Result<String>;
    async fn set_default_style(&self, id: &str) -> Result<()>;
    async fn get_language(&self) -> Result<String>;
    async fn set_language(&self, lang: &str) -> Result<()>;
    async fn get_privacy_mode(&self) -> Result<String>;
    async fn set_privacy_mode(&self, mode: &str) -> Result<()>;
    async fn get_session_history_days(&self) -> Result<i32>;
    async fn set_session_history_days(&self, days: i32) -> Result<()>;
    async fn get_hotkey(&self) -> Result<String>;
    async fn set_hotkey(&self, json: &str) -> Result<()>;
    async fn get_asr_backend(&self) -> Result<String>;
    async fn set_asr_backend(&self, json: &str) -> Result<()>;
    async fn get_llm_backend(&self) -> Result<String>;
    async fn set_llm_backend(&self, json: &str) -> Result<()>;
    async fn get_all_settings(&self) -> Result<String>;
}

#[cold]
#[inline(never)]
fn fatal(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

pub fn run_cli(args: &[String]) {
    // Current-thread runtime: CLI is pure sequential I/O, no benefit from multi-thread.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime for CLI");

    if let Err(e) = rt.block_on(dispatch(args)) {
        fatal(e);
    }
}

async fn dispatch(args: &[String]) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Establish session-bus connection once; reused for proxy.
    let conn = Connection::session()
        .await
        .map_err(|e| format!("failed to connect to session bus: {e}"))?;

    let proxy = DaemonProxy::new(&conn)
        .await
        .map_err(|e| format!("failed to create proxy: {e}"))?;

    // Use byte comparison on first arg to avoid allocation.
    match args.first().map(String::as_str).unwrap_or("") {
        "toggle" => proxy.toggle().await.map_err(daemon_error)?,
        "start" => proxy.start().await.map_err(daemon_error)?,
        "stop" => proxy.stop().await.map_err(daemon_error)?,
        "cancel" => proxy.cancel().await.map_err(daemon_error)?,
        "quit" => proxy.quit().await.map_err(daemon_error)?,
        "status" => {
            // Write directly to stdout lock; avoids per-char formatting overhead.
            let state = proxy.status().await.map_err(daemon_error)?;
            println!("{state}");
        }
        "mode" => handle_mode(&proxy, args.get(1..).unwrap_or(&[])).await?,
        "settings" => handle_settings(&proxy, args.get(1..).unwrap_or(&[])).await?,
        other => {
            eprintln!("unknown subcommand: {other}");
            print_usage();
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn handle_mode(
    proxy: &DaemonProxy<'_>,
    args: &[String],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    match args.first().map(String::as_str) {
        None => {
            // Issue both RPCs concurrently; cuts round-trip latency in half.
            let (dm, am) =
                tokio::try_join!(proxy.get_dictation_mode(), proxy.get_activation_mode(),)
                    .map_err(daemon_error)?;
            println!("dictation: {dm}");
            println!("activation: {am}");
        }
        Some("dictation") => match args.get(1) {
            None => {
                let dm = proxy.get_dictation_mode().await.map_err(daemon_error)?;
                println!("{dm}");
            }
            Some(value) => {
                proxy
                    .set_dictation_mode(value)
                    .await
                    .map_err(daemon_error)?;
            }
        },
        Some("activation") => match args.get(1) {
            None => {
                let am = proxy.get_activation_mode().await.map_err(daemon_error)?;
                println!("{am}");
            }
            Some(value) => {
                proxy
                    .set_activation_mode(value)
                    .await
                    .map_err(daemon_error)?;
            }
        },
        Some(other) => {
            eprintln!("unknown mode subcommand: {other}");
            eprintln!(
                "usage: lune-voice mode [dictation [TypeInPlace|Clipboard|Both] | activation [Toggle|Hold]]"
            );
            std::process::exit(1);
        }
    }
    Ok(())
}

async fn handle_settings(
    proxy: &DaemonProxy<'_>,
    args: &[String],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    match args.first().map(String::as_str) {
        None => {
            let all = proxy.get_all_settings().await.map_err(daemon_error)?;
            println!("{all}");
        }
        Some("style") => match args.get(1) {
            None => {
                let v = proxy.get_default_style().await.map_err(daemon_error)?;
                println!("{v}");
            }
            Some(value) => {
                proxy.set_default_style(value).await.map_err(daemon_error)?;
            }
        },
        Some("language") => match args.get(1) {
            None => {
                let v = proxy.get_language().await.map_err(daemon_error)?;
                println!("{v}");
            }
            Some(value) => {
                proxy.set_language(value).await.map_err(daemon_error)?;
            }
        },
        Some("privacy") => match args.get(1) {
            None => {
                let v = proxy.get_privacy_mode().await.map_err(daemon_error)?;
                println!("{v}");
            }
            Some(value) => {
                proxy.set_privacy_mode(value).await.map_err(daemon_error)?;
            }
        },
        Some("history") => match args.get(1) {
            None => {
                let v = proxy
                    .get_session_history_days()
                    .await
                    .map_err(daemon_error)?;
                if v < 0 {
                    println!("disabled");
                } else {
                    println!("{v}");
                }
            }
            Some(value) => {
                let days: i32 = value
                    .parse()
                    .map_err(|_| format!("invalid number: {value}"))?;
                proxy
                    .set_session_history_days(days)
                    .await
                    .map_err(daemon_error)?;
            }
        },
        Some("hotkey") => match args.get(1) {
            None => {
                let v = proxy.get_hotkey().await.map_err(daemon_error)?;
                println!("{v}");
            }
            Some(value) => {
                proxy.set_hotkey(value).await.map_err(daemon_error)?;
            }
        },
        Some("asr") => match args.get(1) {
            None => {
                let v = proxy.get_asr_backend().await.map_err(daemon_error)?;
                println!("{v}");
            }
            Some(value) => {
                proxy.set_asr_backend(value).await.map_err(daemon_error)?;
            }
        },
        Some("llm") => match args.get(1) {
            None => {
                let v = proxy.get_llm_backend().await.map_err(daemon_error)?;
                println!("{v}");
            }
            Some(value) => {
                proxy.set_llm_backend(value).await.map_err(daemon_error)?;
            }
        },
        Some(other) => {
            eprintln!("unknown settings subcommand: {other}");
            eprintln!(
                "usage: lune-voice settings [style|language|privacy|history|hotkey|asr|llm] [value]"
            );
            std::process::exit(1);
        }
    }
    Ok(())
}

#[inline]
fn daemon_error(e: zbus::Error) -> String {
    // Fast-path: service not running is the common failure; avoid general formatting.
    if let zbus::Error::MethodError(ref name, _, _) = e
        && name.as_str() == "org.freedesktop.DBus.Error.ServiceUnknown"
    {
        return "daemon not running — start with: lune-voice daemon".into();
    }
    format!("D-Bus call failed: {e}")
}

#[cold]
#[inline(never)]
fn print_usage() {
    eprintln!("usage: lune-voice [toggle|start|stop|cancel|status|mode|settings|quit|daemon]");
}
