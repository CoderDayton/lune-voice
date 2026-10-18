// SPDX-License-Identifier: GPL-3.0-or-later

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, SupportedBufferSize};
use tracing::warn;

const PULSE_CAPTURE_PERIOD_MS: u32 = 20;

pub(crate) struct InputDevice {
    pub device: cpal::Device,
    pub names: DeviceNames,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceNames {
    pub name: String,
    pub legacy_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceSelection {
    pub device: Option<usize>,
    pub fell_back: bool,
}

pub fn capture_host() -> cpal::Host {
    cpal::default_host()
}

pub fn playback_host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    if let Ok(host) = cpal::host_from_id(cpal::HostId::Alsa) {
        return host;
    }
    cpal::default_host()
}

#[cfg(target_os = "linux")]
pub(crate) fn is_pulse_host(host: &cpal::Host) -> bool {
    host.id() == cpal::HostId::PulseAudio
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn is_pulse_host(_host: &cpal::Host) -> bool {
    false
}

pub fn capture_buffer_size(
    host: &cpal::Host,
    sample_rate: u32,
    supported: &SupportedBufferSize,
) -> BufferSize {
    period_for_host(is_pulse_host(host), sample_rate, supported)
}

fn period_for_host(pulse: bool, sample_rate: u32, supported: &SupportedBufferSize) -> BufferSize {
    if !pulse {
        return BufferSize::Default;
    }
    let period = (sample_rate * PULSE_CAPTURE_PERIOD_MS / 1000).max(1);
    match *supported {
        SupportedBufferSize::Range { min, max } => {
            BufferSize::Fixed(period.clamp(min, max.max(min)))
        }
        SupportedBufferSize::Unknown => BufferSize::Fixed(period),
    }
}

fn is_microphone_route(pcm_id: &str) -> bool {
    pcm_id.contains("CARD=") && !pcm_id.starts_with("hdmi:") && !pcm_id.starts_with("surround")
}

fn usable_device_indices(described: &[(String, Option<String>)]) -> Vec<usize> {
    fn first_of_each_name<'a>(
        it: impl Iterator<Item = (usize, &'a (String, Option<String>))>,
    ) -> Vec<usize> {
        first_index_per_name(it.map(|(idx, (name, _))| (idx, name.as_str())))
    }

    let microphones = first_of_each_name(
        described
            .iter()
            .enumerate()
            .filter(|(_, (_, pcm))| pcm.as_deref().is_some_and(is_microphone_route)),
    );
    if !microphones.is_empty() {
        return microphones;
    }
    first_of_each_name(described.iter().enumerate())
}

pub(crate) fn first_index_per_name<'a>(it: impl Iterator<Item = (usize, &'a str)>) -> Vec<usize> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    it.filter(|(_, name)| seen.insert(name))
        .map(|(idx, _)| idx)
        .collect()
}

pub(crate) fn enumerate_input_devices(host: &cpal::Host) -> Vec<InputDevice> {
    #[cfg(target_os = "linux")]
    if is_pulse_host(host) {
        match crate::pulse::list_sources() {
            Ok(sources) => {
                let devices = pulse_input_devices(host, &sources);
                if !devices.is_empty() {
                    return devices;
                }
                warn!(
                    "no PulseAudio source matched a cpal input; listing every input cpal reports instead"
                );
            }
            Err(e) => warn!("{e}; listing every input cpal reports instead"),
        }
    }
    cpal_input_devices(host)
}

#[cfg(target_os = "linux")]
fn pulse_input_devices(host: &cpal::Host, sources: &[crate::pulse::Source]) -> Vec<InputDevice> {
    let mut by_id: std::collections::HashMap<String, cpal::Device> = match host.input_devices() {
        Ok(devices) => devices
            .filter_map(|d| Some((d.id().ok()?.id().to_owned(), d)))
            .collect(),
        Err(e) => {
            warn!("enumerating input devices failed: {e}");
            return Vec::new();
        }
    };
    crate::pulse::usable_source_indices(sources, |s| by_id.contains_key(&s.name))
        .into_iter()
        .filter_map(|idx| {
            let source = &sources[idx];
            Some(InputDevice {
                device: by_id.remove(&source.name)?,
                names: DeviceNames {
                    name: source.description.clone(),
                    legacy_names: source.legacy_alsa_name.clone().into_iter().collect(),
                },
            })
        })
        .collect()
}

fn cpal_input_devices(host: &cpal::Host) -> Vec<InputDevice> {
    let devices: Vec<cpal::Device> = match host.input_devices() {
        Ok(devices) => devices.collect(),
        Err(e) => {
            warn!("enumerating input devices failed: {e}");
            return Vec::new();
        }
    };
    let (devices, described): (Vec<cpal::Device>, Vec<(String, Option<String>)>) = devices
        .into_iter()
        .filter_map(|d| {
            let desc = d.description().ok()?;
            let entry = (desc.name().to_owned(), desc.driver().map(str::to_owned));
            Some((d, entry))
        })
        .unzip();

    let mut devices: Vec<Option<cpal::Device>> = devices.into_iter().map(Some).collect();
    usable_device_indices(&described)
        .into_iter()
        .filter_map(|idx| {
            devices[idx].take().map(|device| InputDevice {
                device,
                names: DeviceNames {
                    name: described[idx].0.clone(),
                    legacy_names: Vec::new(),
                },
            })
        })
        .collect()
}

pub fn list_input_devices() -> Vec<String> {
    enumerate_input_devices(&capture_host())
        .into_iter()
        .map(|d| d.names.name)
        .collect()
}

pub fn migrated_input_device_name(saved: &str) -> Option<String> {
    if !could_be_legacy_name(saved) {
        return None;
    }
    let names: Vec<DeviceNames> = enumerate_input_devices(&capture_host())
        .into_iter()
        .map(|d| d.names)
        .collect();
    migrated_name(saved, &names)
}

fn could_be_legacy_name(saved: &str) -> bool {
    saved.contains(", ")
}

fn migrated_name(saved: &str, available: &[DeviceNames]) -> Option<String> {
    let idx = resolve_input_device(Some(saved), available).device?;
    let current = &available[idx].name;
    (current != saved).then(|| current.clone())
}

pub(crate) fn resolve_input_device(
    requested: Option<&str>,
    available: &[DeviceNames],
) -> DeviceSelection {
    let Some(requested) = requested else {
        return DeviceSelection {
            device: None,
            fell_back: false,
        };
    };
    let device = available
        .iter()
        .position(|d| d.name == requested)
        .or_else(|| {
            available
                .iter()
                .position(|d| d.legacy_names.iter().any(|legacy| legacy == requested))
        });
    DeviceSelection {
        fell_back: device.is_none(),
        device,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE_48K: u32 = 48_000;

    fn names(name: &str, legacy: &[&str]) -> DeviceNames {
        DeviceNames {
            name: name.into(),
            legacy_names: legacy.iter().map(|s| (*s).into()).collect(),
        }
    }

    #[test]
    fn resolve_input_device_falls_back_when_requested_missing() {
        let available = [names("Built-in Microphone", &[]), names("HDMI", &[])];
        assert_eq!(
            resolve_input_device(Some("USB Headset"), &available),
            DeviceSelection {
                device: None,
                fell_back: true,
            }
        );
    }

    #[test]
    fn resolve_input_device_selects_requested_when_present() {
        let available = [names("Built-in Microphone", &[]), names("USB Headset", &[])];
        assert_eq!(
            resolve_input_device(Some("USB Headset"), &available),
            DeviceSelection {
                device: Some(1),
                fell_back: false,
            }
        );
    }

    #[test]
    fn resolve_input_device_uses_default_when_none_requested() {
        let available = [names("Built-in Microphone", &[])];
        assert_eq!(
            resolve_input_device(None, &available),
            DeviceSelection {
                device: None,
                fell_back: false,
            }
        );
    }

    #[test]
    fn resolve_input_device_matches_a_legacy_alsa_name() {
        let available = [
            names("DeepFilter Shure MV6", &[]),
            names("Shure MV6 Mono", &["Shure MV6, USB Audio"]),
        ];
        assert_eq!(
            resolve_input_device(Some("Shure MV6, USB Audio"), &available),
            DeviceSelection {
                device: Some(1),
                fell_back: false,
            }
        );
    }

    #[test]
    fn resolve_input_device_prefers_a_current_name_over_a_legacy_one() {
        let available = [names("Headset", &["USB Mic"]), names("USB Mic", &[])];
        assert_eq!(
            resolve_input_device(Some("USB Mic"), &available).device,
            Some(1)
        );
    }

    #[test]
    fn migrated_name_maps_a_legacy_name_to_the_current_one() {
        let available = [names("Shure MV6 Mono", &["Shure MV6, USB Audio"])];
        assert_eq!(
            migrated_name("Shure MV6, USB Audio", &available).as_deref(),
            Some("Shure MV6 Mono")
        );
    }

    #[test]
    fn only_an_alsa_shaped_name_needs_the_device_list() {
        assert!(could_be_legacy_name("Shure MV6, USB Audio"));
        assert!(could_be_legacy_name("HDA Intel PCH, ALC1220 Analog"));
        assert!(!could_be_legacy_name("Shure MV6 Mono"));
        assert!(!could_be_legacy_name("DeepFilter Shure MV6"));
        assert!(!could_be_legacy_name(""));
    }

    #[test]
    fn migrated_name_leaves_current_and_unknown_names_alone() {
        let available = [names("Shure MV6 Mono", &["Shure MV6, USB Audio"])];
        assert_eq!(migrated_name("Shure MV6 Mono", &available), None);
        assert_eq!(migrated_name("Unplugged Mic", &available), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn playback_stays_on_alsa() {
        assert_eq!(playback_host().id(), cpal::HostId::Alsa);
    }

    #[test]
    fn capture_buffer_size_is_left_to_the_host_off_pulse() {
        let range = SupportedBufferSize::Range {
            min: 1,
            max: 1 << 20,
        };
        assert_eq!(
            period_for_host(false, RATE_48K, &range),
            BufferSize::Default
        );
    }

    #[test]
    fn capture_buffer_size_requests_a_short_period_on_pulse() {
        let range = SupportedBufferSize::Range {
            min: 1,
            max: 1 << 20,
        };
        assert_eq!(
            period_for_host(true, RATE_48K, &range),
            BufferSize::Fixed(960)
        );
        assert_eq!(
            period_for_host(true, 44_100, &range),
            BufferSize::Fixed(882)
        );
        assert_eq!(
            period_for_host(true, RATE_48K, &SupportedBufferSize::Unknown),
            BufferSize::Fixed(960)
        );
    }

    #[test]
    fn capture_buffer_size_stays_inside_the_supported_range() {
        let range = SupportedBufferSize::Range {
            min: 2048,
            max: 4096,
        };
        assert_eq!(
            period_for_host(true, RATE_48K, &range),
            BufferSize::Fixed(2048)
        );
        let range = SupportedBufferSize::Range { min: 1, max: 256 };
        assert_eq!(
            period_for_host(true, RATE_48K, &range),
            BufferSize::Fixed(256)
        );
    }

    fn described(entries: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
        entries
            .iter()
            .map(|(pcm, name)| ((*name).to_string(), Some((*pcm).to_string())))
            .collect()
    }

    fn kept(entries: &[(&str, &str)]) -> Vec<String> {
        usable_device_indices(&described(entries))
            .into_iter()
            .map(|idx| entries[idx].1.to_string())
            .collect()
    }

    const REAL_ALSA_ENUMERATION: &[(&str, &str)] = &[
        (
            "null",
            "Discard all samples (playback) or generate zero samples (capture)",
        ),
        ("pipewire", "PipeWire Sound Server"),
        (
            "default",
            "Default ALSA Output (currently PipeWire Media Server)",
        ),
        ("hw:CARD=PCH,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
        ("hw:CARD=PCH,DEV=2", "HDA Intel PCH, ALC1220 Alt Analog"),
        ("plughw:CARD=PCH,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
        ("plughw:CARD=PCH,DEV=2", "HDA Intel PCH, ALC1220 Alt Analog"),
        ("sysdefault:CARD=PCH", "HDA Intel PCH, ALC1220 Analog"),
        ("front:CARD=PCH,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
        ("dsnoop:CARD=PCH,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
        ("dsnoop:CARD=PCH,DEV=2", "HDA Intel PCH, ALC1220 Alt Analog"),
        ("hw:CARD=Audio,DEV=0", "DigiHug USB Audio, USB Audio"),
        ("plughw:CARD=Audio,DEV=0", "DigiHug USB Audio, USB Audio"),
        ("sysdefault:CARD=Audio", "DigiHug USB Audio, USB Audio"),
        ("front:CARD=Audio,DEV=0", "DigiHug USB Audio, USB Audio"),
        ("dsnoop:CARD=Audio,DEV=0", "DigiHug USB Audio, USB Audio"),
        ("hw:CARD=MV6,DEV=0", "Shure MV6, USB Audio"),
        ("plughw:CARD=MV6,DEV=0", "Shure MV6, USB Audio"),
        ("sysdefault:CARD=MV6", "Shure MV6, USB Audio"),
        ("front:CARD=MV6,DEV=0", "Shure MV6, USB Audio"),
        ("dsnoop:CARD=MV6,DEV=0", "Shure MV6, USB Audio"),
        ("hw:CARD=0,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
        ("plughw:CARD=0,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
        ("hw:CARD=0,DEV=2", "HDA Intel PCH, ALC1220 Alt Analog"),
        ("plughw:CARD=0,DEV=2", "HDA Intel PCH, ALC1220 Alt Analog"),
        ("hw:CARD=1,DEV=0", "DigiHug USB Audio, USB Audio"),
        ("plughw:CARD=1,DEV=0", "DigiHug USB Audio, USB Audio"),
        ("hw:CARD=3,DEV=0", "Shure MV6, USB Audio"),
        ("plughw:CARD=3,DEV=0", "Shure MV6, USB Audio"),
    ];

    #[test]
    fn usable_devices_reduces_real_alsa_enumeration_to_one_entry_per_mic() {
        assert_eq!(REAL_ALSA_ENUMERATION.len(), 29);
        assert_eq!(
            kept(REAL_ALSA_ENUMERATION),
            [
                "HDA Intel PCH, ALC1220 Analog",
                "HDA Intel PCH, ALC1220 Alt Analog",
                "DigiHug USB Audio, USB Audio",
                "Shure MV6, USB Audio",
            ]
        );
    }

    #[test]
    fn usable_devices_keeps_the_first_route_for_a_name() {
        let indices = usable_device_indices(&described(REAL_ALSA_ENUMERATION));
        assert_eq!(
            indices
                .iter()
                .map(|&i| REAL_ALSA_ENUMERATION[i].0)
                .collect::<Vec<_>>(),
            [
                "hw:CARD=PCH,DEV=0",
                "hw:CARD=PCH,DEV=2",
                "hw:CARD=Audio,DEV=0",
                "hw:CARD=MV6,DEV=0",
            ]
        );
    }

    #[test]
    fn usable_devices_drops_speaker_only_routes() {
        assert_eq!(
            kept(&[
                ("hdmi:CARD=PCH,DEV=0", "HDA Intel PCH, HDMI 0"),
                ("surround51:CARD=PCH,DEV=0", "HDA Intel PCH, ALC1220 Analog"),
                ("hw:CARD=MV6,DEV=0", "Shure MV6, USB Audio"),
            ]),
            ["Shure MV6, USB Audio"]
        );
    }

    #[test]
    fn usable_devices_fall_back_when_no_card_backed_route_exists() {
        assert_eq!(
            kept(&[
                ("pipewire", "PipeWire Sound Server"),
                ("default", "Default ALSA Output"),
                ("pipewire", "PipeWire Sound Server"),
            ]),
            ["PipeWire Sound Server", "Default ALSA Output"]
        );
    }

    #[test]
    fn usable_devices_keep_non_alsa_hosts_intact() {
        assert_eq!(
            kept(&[
                ("CoreAudio", "MacBook Pro Microphone"),
                ("CoreAudio", "Shure MV6"),
            ]),
            ["MacBook Pro Microphone", "Shure MV6"]
        );
    }
}
