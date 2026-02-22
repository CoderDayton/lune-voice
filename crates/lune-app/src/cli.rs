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
}

pub fn run_cli(args: &[String]) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime for CLI");

    if let Err(e) = rt.block_on(dispatch(args)) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn dispatch(args: &[String]) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::session()
        .await
        .map_err(|e| format!("failed to connect to session bus: {e}"))?;

    let proxy = DaemonProxy::new(&conn)
        .await
        .map_err(|e| format!("failed to create proxy: {e}"))?;

    let subcommand = args.first().map(|s| s.as_str()).unwrap_or("");

    match subcommand {
        "toggle" => {
            proxy.toggle().await.map_err(daemon_error)?;
        }
        "start" => {
            proxy.start().await.map_err(daemon_error)?;
        }
        "stop" => {
            proxy.stop().await.map_err(daemon_error)?;
        }
        "cancel" => {
            proxy.cancel().await.map_err(daemon_error)?;
        }
        "quit" => {
            proxy.quit().await.map_err(daemon_error)?;
        }
        "status" => {
            let state = proxy.status().await.map_err(daemon_error)?;
            println!("{state}");
        }
        "mode" => {
            handle_mode(&proxy, &args[1..]).await?;
        }
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
    match args.first().map(|s| s.as_str()) {
        None => {
            // Print both modes.
            let dm = proxy.get_dictation_mode().await.map_err(daemon_error)?;
            let am = proxy.get_activation_mode().await.map_err(daemon_error)?;
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

fn daemon_error(e: zbus::Error) -> String {
    if let zbus::Error::MethodError(ref name, _, _) = e
        && name.as_str() == "org.freedesktop.DBus.Error.ServiceUnknown"
    {
        return "daemon not running — start with: lune-voice daemon".into();
    }
    format!("D-Bus call failed: {e}")
}

fn print_usage() {
    eprintln!("usage: lune-voice [toggle|start|stop|cancel|status|mode|quit|daemon]");
}
