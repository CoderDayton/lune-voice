// SPDX-License-Identifier: GPL-3.0-or-later

use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::time::Duration;

const LIST_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Source {
    pub name: String,
    pub description: String,
    pub is_monitor: bool,
    pub legacy_alsa_name: Option<String>,
}

pub(crate) fn list_sources() -> Result<Vec<Source>, String> {
    let path = pulseaudio::socket_path_from_env()
        .ok_or_else(|| "no PulseAudio server socket found".to_owned())?;
    let cookie = pulseaudio::cookie_path_from_env().and_then(|p| std::fs::read(p).ok());
    list_sources_at(&path, cookie, LIST_TIMEOUT)
}

fn list_sources_at(
    path: &std::path::Path,
    cookie: Option<Vec<u8>>,
    timeout: Duration,
) -> Result<Vec<Source>, String> {
    let socket = UnixStream::connect(path).map_err(|e| format!("connect to PulseAudio: {e}"))?;
    let guard = socket
        .try_clone()
        .map_err(|e| format!("clone PulseAudio socket: {e}"))?;
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("lune-pulse-list".into())
        .spawn(move || {
            let _ = tx.send(query_sources(socket, cookie));
        })
        .map_err(|e| format!("spawn PulseAudio listing thread: {e}"))?;
    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => {
            let _ = guard.shutdown(Shutdown::Both);
            Err(format!(
                "PulseAudio server did not answer within {timeout:?}"
            ))
        }
    }
}

fn query_sources(socket: UnixStream, cookie: Option<Vec<u8>>) -> Result<Vec<Source>, String> {
    let client = pulseaudio::Client::new_unix(c"lune-voice", socket, cookie)
        .map_err(|e| format!("connect to PulseAudio: {e}"))?;
    let infos = futures_executor::block_on(client.list_sources())
        .map_err(|e| format!("list PulseAudio sources: {e}"))?;
    Ok(infos.iter().map(source_from_info).collect())
}

fn source_from_info(info: &pulseaudio::protocol::SourceInfo) -> Source {
    let prop = |key: &str| {
        info.props
            .iter()
            .find(|(k, _)| k.to_bytes() == key.as_bytes())
            .map(|(_, v)| String::from_utf8_lossy(v).trim_end_matches('\0').to_owned())
    };
    let name = info.name.to_string_lossy().into_owned();
    Source {
        description: info
            .description
            .as_ref()
            .map_or_else(|| name.clone(), |d| d.to_string_lossy().into_owned()),
        is_monitor: info.monitor_of_sink_index.is_some(),
        legacy_alsa_name: legacy_alsa_name(
            prop("alsa.card_name").as_deref(),
            prop("alsa.name").as_deref(),
        ),
        name,
    }
}

pub(crate) fn legacy_alsa_name(card_name: Option<&str>, pcm_name: Option<&str>) -> Option<String> {
    let (card, pcm) = (card_name?, pcm_name?);
    (!card.is_empty() && !pcm.is_empty()).then(|| format!("{card}, {pcm}"))
}

pub(crate) fn usable_source_indices(
    sources: &[Source],
    openable: impl Fn(&Source) -> bool,
) -> Vec<usize> {
    crate::devices::first_index_per_name(
        sources
            .iter()
            .enumerate()
            .filter(|(_, source)| !source.is_monitor && openable(source))
            .map(|(idx, source)| (idx, source.description.as_str())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_server_that_never_answers_times_out_and_releases_the_connection() {
        use std::io::Read;
        use std::os::unix::net::UnixListener;
        use std::time::Instant;

        let path = std::env::temp_dir().join(format!(
            "lune-pulse-silent-{}-{:?}.sock",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let started = Instant::now();
        let result = list_sources_at(&path, None, Duration::from_millis(300));
        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(2));

        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut server_side = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => panic!("the client never connected: {e}"),
            }
        };
        server_side.set_nonblocking(false).unwrap();
        server_side
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut buf = [0u8; 4096];
        let closed = loop {
            match server_side.read(&mut buf) {
                Ok(0) => break true,
                Ok(_) => continue,
                Err(_) => break false,
            }
        };
        let _ = std::fs::remove_file(&path);
        assert!(closed, "the client socket was left open");
    }

    fn source(name: &str, description: &str, is_monitor: bool, legacy: Option<&str>) -> Source {
        Source {
            name: name.into(),
            description: description.into(),
            is_monitor,
            legacy_alsa_name: legacy.map(Into::into),
        }
    }

    fn real_pipewire_pulse_sources() -> Vec<Source> {
        vec![
            source("deepfilter_source_mv6", "DeepFilter Shure MV6", false, None),
            source(
                "alsa_output.usb-Shure_Inc_Shure_MV6_MV6_6-01.analog-stereo.monitor",
                "Monitor of Shure MV6 Analog Stereo",
                true,
                Some("Shure MV6, USB Audio"),
            ),
            source(
                "alsa_input.usb-Shure_Inc_Shure_MV6_MV6_6-01.mono-fallback",
                "Shure MV6 Mono",
                false,
                Some("Shure MV6, USB Audio"),
            ),
            source(
                "alsa_output.usb-FiiO_DigiHug_USB_Audio-01.analog-stereo.monitor",
                "Monitor of Fiio E10 Analog Stereo",
                true,
                Some("DigiHug USB Audio, USB Audio"),
            ),
            source(
                "alsa_input.usb-FiiO_DigiHug_USB_Audio-01.analog-stereo",
                "Fiio E10 Analog Stereo",
                false,
                Some("DigiHug USB Audio, USB Audio"),
            ),
        ]
    }

    #[test]
    fn usable_sources_drop_sink_monitors() {
        let sources = real_pipewire_pulse_sources();
        let kept: Vec<&str> = usable_source_indices(&sources, |_| true)
            .into_iter()
            .map(|i| sources[i].name.as_str())
            .collect();
        assert_eq!(
            kept,
            [
                "deepfilter_source_mv6",
                "alsa_input.usb-Shure_Inc_Shure_MV6_MV6_6-01.mono-fallback",
                "alsa_input.usb-FiiO_DigiHug_USB_Audio-01.analog-stereo",
            ]
        );
    }

    #[test]
    fn usable_sources_keep_the_first_source_for_a_shared_description() {
        let sources = vec![
            source("a", "USB Mic", false, None),
            source("b", "USB Mic", false, None),
        ];
        assert_eq!(usable_source_indices(&sources, |_| true), [0]);
    }

    #[test]
    fn usable_sources_skip_a_source_cpal_cannot_open_before_deduplicating() {
        let sources = vec![
            source("gone", "USB Mic", false, None),
            source("b", "USB Mic", false, None),
        ];
        assert_eq!(usable_source_indices(&sources, |s| s.name != "gone"), [1]);
    }

    #[test]
    fn legacy_alsa_name_joins_card_and_pcm_names() {
        assert_eq!(
            legacy_alsa_name(Some("Shure MV6"), Some("USB Audio")).as_deref(),
            Some("Shure MV6, USB Audio")
        );
    }

    #[test]
    fn legacy_alsa_name_needs_both_parts() {
        assert_eq!(legacy_alsa_name(None, Some("USB Audio")), None);
        assert_eq!(legacy_alsa_name(Some("Shure MV6"), None), None);
        assert_eq!(legacy_alsa_name(Some(""), Some("USB Audio")), None);
        assert_eq!(legacy_alsa_name(Some("Shure MV6"), Some("")), None);
    }
}
