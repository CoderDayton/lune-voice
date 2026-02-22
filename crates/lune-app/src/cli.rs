use zbus::{Connection, Result, proxy};

#[proxy(
    interface = "com.lunevoice.Daemon1",
    default_service = "com.lunevoice.Daemon1",
    default_path = "/com/lunevoice/Daemon1"
)]
trait Daemon {
    async fn toggle(&self) -> Result<()>;
    async fn cancel(&self) -> Result<()>;
    async fn quit(&self) -> Result<()>;
    async fn status(&self) -> Result<String>;
}

pub fn run_cli(subcommand: &str) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime for CLI");

    if let Err(e) = rt.block_on(dispatch(subcommand)) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn dispatch(subcommand: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::session()
        .await
        .map_err(|e| format!("failed to connect to session bus: {e}"))?;

    let proxy = DaemonProxy::new(&conn)
        .await
        .map_err(|e| format!("failed to create proxy: {e}"))?;

    match subcommand {
        "toggle" => {
            proxy.toggle().await.map_err(daemon_error)?;
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
        other => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("usage: lune-voice [toggle|cancel|status|quit|daemon]");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn daemon_error(e: zbus::Error) -> String {
    if let zbus::Error::MethodError(ref name, _, _) = e {
        if name.as_str() == "org.freedesktop.DBus.Error.ServiceUnknown" {
            return "daemon not running — start with: lune-voice daemon".into();
        }
    }
    format!("D-Bus call failed: {e}")
}
