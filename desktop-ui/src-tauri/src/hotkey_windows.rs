// SPDX-License-Identifier: GPL-3.0-or-later

//! Windows global hotkey detection via a low-level keyboard hook.
//!
//! `tauri-plugin-global-shortcut` cannot serve this app on Windows: it reports
//! a single "activated" callback, so there is no key *release* to end a
//! push-to-talk hold on. `WH_KEYBOARD_LL` sees raw down/up transitions for
//! every key on the machine, which is the only Windows API that can express
//! both activation modes.
//!
//! Unlike the Linux backend this does **not** take the keyboard away from the
//! rest of the desktop. The hook observes the stream and forwards everything
//! it did not match, so there is no passthrough device to maintain and no way
//! for a crash here to freeze the machine's input.
//!
//! The decision core below is deliberately free of Win32 types and compiles on
//! every platform, so the activation-mode state machine is unit-tested on the
//! build host. Only `start` and the hook thread are Windows-gated.

use lune_core::config::{ActivationMode, HotkeyConfig, Modifier};

use crate::hotkey::HotkeyAction;

// ─── Virtual-key codes ───────────────────────────────────────────────────────
//
// Spelled out rather than pulled from `windows-sys` so the decision core stays
// portable. These are stable ABI constants; they cannot drift.

/// A Windows virtual-key code. Only 0..=254 are assignable.
pub(crate) type Vk = u16;

const VK_LSHIFT: Vk = 0xA0;
const VK_RSHIFT: Vk = 0xA1;
const VK_LCONTROL: Vk = 0xA2;
const VK_RCONTROL: Vk = 0xA3;
const VK_LMENU: Vk = 0xA4;
const VK_RMENU: Vk = 0xA5;
const VK_LWIN: Vk = 0x5B;
const VK_RWIN: Vk = 0x5C;

/// Sentinel for "no cancel key bound", mirroring the Linux backend's use of
/// `KEY_RESERVED`. `0x00` is not an assignable virtual-key code.
const VK_NONE: Vk = 0x00;

/// The two physical keys that count as one modifier. A binding says "Ctrl",
/// never "left Ctrl", so either side satisfies it and either side alone is
/// enough to make an unbound modifier "extra".
const fn modifier_keys(m: Modifier) -> Option<(Vk, Vk)> {
    Some(match m {
        Modifier::Ctrl => (VK_LCONTROL, VK_RCONTROL),
        Modifier::Shift => (VK_LSHIFT, VK_RSHIFT),
        Modifier::Alt => (VK_LMENU, VK_RMENU),
        Modifier::Meta => (VK_LWIN, VK_RWIN),
        // `Modifier` is #[non_exhaustive]; an unknown one binds nothing rather
        // than silently matching everything.
        _ => return None,
    })
}

/// Every modifier family, for the "no extra modifiers held" check.
const ALL_MODIFIERS: [Modifier; 4] = [
    Modifier::Ctrl,
    Modifier::Shift,
    Modifier::Alt,
    Modifier::Meta,
];

/// Map a config key name to a virtual-key code.
///
/// Accepts exactly the names `hotkey_linux::key_name_to_code` accepts, so one
/// `settings.toml` describes the same binding on both platforms.
pub(crate) fn key_name_to_vk(name: &str) -> Option<Vk> {
    let lower = name.to_ascii_lowercase();
    Some(match lower.as_str() {
        "space" => 0x20,
        "escape" | "esc" => 0x1B,
        "return" | "enter" => 0x0D,
        "tab" => 0x09,
        "backspace" => 0x08,
        "delete" => 0x2E,
        "insert" => 0x2D,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        "up" => 0x26,
        "down" => 0x28,
        "left" => 0x25,
        "right" => 0x27,
        "capslock" => 0x14,
        "pause" => 0x13,
        "printscreen" | "sysrq" => 0x2C,
        "minus" => 0xBD,
        "equal" | "equals" => 0xBB,
        "leftbrace" => 0xDB,
        "rightbrace" => 0xDD,
        "backslash" => 0xDC,
        "semicolon" => 0xBA,
        "apostrophe" | "quote" => 0xDE,
        "grave" | "backtick" => 0xC0,
        "comma" => 0xBC,
        "dot" | "period" => 0xBE,
        "slash" => 0xBF,
        _ => alphanumeric_or_function_key(&lower)?,
    })
}

/// The three name families that map by rule rather than by table.
fn alphanumeric_or_function_key(lower: &str) -> Option<Vk> {
    let mut chars = lower.chars();
    let (c, rest) = (chars.next()?, chars.as_str());
    match c {
        // VK_A..VK_Z and VK_0..VK_9 are the ASCII values.
        'a'..='z' if rest.is_empty() => Some(c.to_ascii_uppercase() as Vk),
        '0'..='9' if rest.is_empty() => Some(c as Vk),
        // F1..F12 are contiguous from VK_F1. Anything past F12 is unbindable
        // here, and `ff` must not parse as a function key at all.
        'f' => rest
            .parse::<u16>()
            .ok()
            .filter(|n| (1..=12).contains(n))
            .map(|n| 0x70 + (n - 1)),
        _ => None,
    }
}

// ─── Pressed-key set ─────────────────────────────────────────────────────────

/// Which virtual keys are physically down. Fixed 256-bit set: the virtual-key
/// space is exactly one byte, so this can never need to grow or allocate — and
/// the hook callback must not allocate.
#[derive(Default, Clone, Copy)]
pub(crate) struct PressedVks {
    w: [u64; 4],
}

impl PressedVks {
    fn insert(&mut self, vk: Vk) {
        if let Some(i) = Self::index(vk) {
            self.w[i / 64] |= 1u64 << (i % 64);
        }
    }

    fn remove(&mut self, vk: Vk) {
        if let Some(i) = Self::index(vk) {
            self.w[i / 64] &= !(1u64 << (i % 64));
        }
    }

    fn contains(&self, vk: Vk) -> bool {
        Self::index(vk).is_some_and(|i| (self.w[i / 64] >> (i % 64)) & 1 != 0)
    }

    const fn index(vk: Vk) -> Option<usize> {
        if vk < 256 { Some(vk as usize) } else { None }
    }
}

// ─── Resolved binding ────────────────────────────────────────────────────────

/// A `HotkeyConfig` resolved to virtual-key codes. The parse happens once, off
/// the hook thread, so a key event never touches a string.
#[derive(Clone)]
pub(crate) struct WinSnapshot {
    main_trigger: Vk,
    main_mods: Vec<Modifier>,
    cancel_trigger: Vk,
    cancel_mods: Vec<Modifier>,
    mode: ActivationMode,
}

impl WinSnapshot {
    /// `None` when the *toggle* key name is unrecognised — that one is
    /// load-bearing. An unrecognised cancel key only disables cancel.
    pub(crate) fn from_config(config: &HotkeyConfig, mode: ActivationMode) -> Option<Self> {
        let main_trigger = key_name_to_vk(&config.toggle_dictation.key)?;
        let cancel_trigger = key_name_to_vk(&config.cancel_dictation.key).unwrap_or_else(|| {
            tracing::warn!(
                key = %config.cancel_dictation.key,
                "unknown cancel hotkey key; cancel hotkey disabled"
            );
            VK_NONE
        });
        Some(Self {
            main_trigger,
            main_mods: config.toggle_dictation.modifiers.clone(),
            cancel_trigger,
            cancel_mods: config.cancel_dictation.modifiers.clone(),
            mode,
        })
    }
}

// ─── Decision core ───────────────────────────────────────────────────────────

/// What the hook should do with the key it just saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Decision {
    /// Intent to publish, if this transition completed one.
    pub(crate) action: Option<HotkeyAction>,
    /// Whether to withhold this key from the rest of the desktop.
    ///
    /// Only the matched trigger is swallowed, and only while its modifiers are
    /// satisfied — otherwise binding a bare letter would make that letter
    /// untypeable everywhere.
    pub(crate) swallow: bool,
}

/// The activation-mode state machine, one instance owned by the hook thread.
pub(crate) struct WinHotkeyState {
    pressed: PressedVks,
    snap: WinSnapshot,
    hold_active: bool,
    /// Whether the trigger's *current* press was withheld from the desktop.
    ///
    /// The release has to follow the press it belongs to. Deciding it afresh
    /// from the modifiers held at release time gets it wrong whenever they
    /// changed in between — a Space forwarded to the app, then Ctrl pressed,
    /// would have its release swallowed and leave the key wedged down there.
    trigger_swallowed: bool,
    /// Set when `Cancel` ended a hold whose trigger is still physically down.
    ///
    /// Holds the machine out of the hold state until the combo is actually
    /// released, so the eventual release cannot emit a `Stop` for a session
    /// cancel already ended, and key repeat on the still-held trigger cannot
    /// restart one.
    hold_suppressed: bool,
}

impl WinHotkeyState {
    pub(crate) fn new(snap: WinSnapshot) -> Self {
        Self {
            pressed: PressedVks::default(),
            snap,
            hold_active: false,
            trigger_swallowed: false,
            hold_suppressed: false,
        }
    }

    /// Swap the binding in place, preserving the physical pressed-key set.
    ///
    /// `hold_active` resets: a hold in progress under the old combo must not
    /// emit a `Stop` for the new one.
    pub(crate) fn retarget(&mut self, snap: WinSnapshot) {
        self.snap = snap;
        self.hold_active = false;
        // The old trigger's release is forwarded anyway once the binding no
        // longer names it, so nothing is owed to the desktop here.
        self.trigger_swallowed = false;
        self.hold_suppressed = false;
    }

    /// Feed one key transition. `pressed` is false for a key-up.
    pub(crate) fn handle(&mut self, vk: Vk, pressed: bool) -> Decision {
        let was_trigger = self.pressed.contains(self.snap.main_trigger);
        let was_hold = self.hold_active;

        if pressed {
            self.pressed.insert(vk);
        } else {
            self.pressed.remove(vk);
        }

        // Cancel (Escape) in any activation mode. A session can be started by
        // clicking the pill — which has no key to release — so Escape must be
        // able to stop it even in Hold mode. Emitting Cancel when idle is
        // harmless: the dispatcher no-ops unless a session is active.
        //
        // Never swallowed. Escape is load-bearing in every other app, and this
        // hook sees the whole machine's keyboard.
        if pressed
            && vk == self.snap.cancel_trigger
            && self.snap.cancel_trigger != VK_NONE
            && self.mods_exact(&self.snap.cancel_mods)
        {
            // Cancel ends the session, so any hold feeding it is over too.
            // Leaving `hold_active` set would emit a `Stop` for a session that
            // no longer exists once the trigger is let go, and key repeat on
            // the still-held trigger would start a fresh one. Latch instead;
            // the latch lifts when the combo is genuinely released.
            self.hold_suppressed = self.hold_active;
            self.hold_active = false;
            return Decision {
                action: Some(HotkeyAction::Cancel),
                swallow: false,
            };
        }

        let trigger = self.pressed.contains(self.snap.main_trigger);
        let exact = self.mods_exact(&self.snap.main_mods);
        // Press and release are one decision, taken on the press edge and
        // remembered. Swallowing only the release of a press the desktop
        // already saw would strand that key down in the foreground window;
        // forwarding only the release of a press we consumed does the same in
        // reverse. Key repeat re-delivers the down transition and must inherit
        // the original answer rather than re-derive it.
        let swallow = if vk == self.snap.main_trigger {
            if pressed {
                if !was_trigger {
                    self.trigger_swallowed = exact;
                }
                self.trigger_swallowed
            } else {
                std::mem::take(&mut self.trigger_swallowed)
            }
        } else {
            false
        };

        let action = match self.snap.mode {
            ActivationMode::Hold => {
                let engaged = trigger && exact;
                if !engaged {
                    self.hold_suppressed = false;
                }
                self.hold_active = engaged && !self.hold_suppressed;
                match (was_hold, self.hold_active) {
                    (false, true) => Some(HotkeyAction::Start),
                    (true, false) => Some(HotkeyAction::Stop),
                    _ => None,
                }
            }
            // Toggle, and any mode added later: fire once on the press edge.
            _ => (!was_trigger && trigger && exact).then_some(HotkeyAction::Toggle),
        };

        Decision { action, swallow }
    }

    /// Every required modifier is down and no unrequired one is.
    fn mods_exact(&self, required: &[Modifier]) -> bool {
        ALL_MODIFIERS.iter().all(|m| {
            let Some((l, r)) = modifier_keys(*m) else {
                return true;
            };
            let down = self.pressed.contains(l) || self.pressed.contains(r);
            down == required.contains(m)
        })
    }
}

// ─── Windows hook thread ─────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
mod hook;

/// Install the low-level keyboard hook on a dedicated thread.
///
/// Returns `None` if the binding cannot be resolved or the thread will not
/// start; the caller then falls back to the global-shortcut plugin.
#[cfg(target_os = "windows")]
pub fn start(
    config: &HotkeyConfig,
    mode: ActivationMode,
    sink: crate::hotkey::HotkeySink,
) -> Option<std::sync::Arc<dyn crate::hotkey::HotkeyBackend>> {
    hook::start(config, mode, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::{HotkeyConfig, KeyCombo};

    const VK_SPACE: Vk = 0x20;
    const VK_ESCAPE: Vk = 0x1B;

    fn config(toggle_mods: Vec<Modifier>, toggle_key: &str) -> HotkeyConfig {
        HotkeyConfig {
            toggle_dictation: KeyCombo {
                modifiers: toggle_mods,
                key: toggle_key.into(),
            },
            cancel_dictation: KeyCombo {
                modifiers: vec![],
                key: "escape".into(),
            },
        }
    }

    fn state(mods: Vec<Modifier>, key: &str, mode: ActivationMode) -> WinHotkeyState {
        WinHotkeyState::new(WinSnapshot::from_config(&config(mods, key), mode).unwrap())
    }

    #[test]
    fn letters_digits_and_function_keys_map_to_their_vk_codes() {
        assert_eq!(key_name_to_vk("a"), Some(0x41));
        assert_eq!(key_name_to_vk("Z"), Some(0x5A));
        assert_eq!(key_name_to_vk("0"), Some(0x30));
        assert_eq!(key_name_to_vk("9"), Some(0x39));
        assert_eq!(key_name_to_vk("f1"), Some(0x70));
        assert_eq!(key_name_to_vk("f12"), Some(0x7B));
        assert_eq!(key_name_to_vk("space"), Some(0x20));
        assert_eq!(key_name_to_vk("esc"), Some(0x1B));
    }

    /// `f13` and `ff` both start with 'f' and must not fall into the F-key arm.
    #[test]
    fn unknown_key_names_are_rejected_rather_than_guessed() {
        assert_eq!(key_name_to_vk("f13"), None);
        assert_eq!(key_name_to_vk("f0"), None);
        assert_eq!(key_name_to_vk("ff"), None);
        assert_eq!(key_name_to_vk("hyper"), None);
        assert_eq!(key_name_to_vk(""), None);
    }

    #[test]
    fn toggle_mode_fires_once_on_the_press_edge() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);

        assert_eq!(s.handle(VK_LCONTROL, true).action, None);
        assert_eq!(
            s.handle(VK_SPACE, true).action,
            Some(HotkeyAction::Toggle),
            "press with the modifier held should fire"
        );
        // Key repeat re-delivers the same down transition; it must not re-fire.
        assert_eq!(s.handle(VK_SPACE, true).action, None);
        assert_eq!(s.handle(VK_SPACE, false).action, None, "release is silent");
    }

    #[test]
    fn hold_mode_starts_on_press_and_stops_on_release() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Hold);

        s.handle(VK_LCONTROL, true);
        assert_eq!(s.handle(VK_SPACE, true).action, Some(HotkeyAction::Start));
        assert_eq!(s.handle(VK_SPACE, false).action, Some(HotkeyAction::Stop));
    }

    /// Letting go of the modifier first is the common way a hold ends.
    #[test]
    fn hold_mode_stops_when_the_modifier_is_released_first() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Hold);

        s.handle(VK_LCONTROL, true);
        assert_eq!(s.handle(VK_SPACE, true).action, Some(HotkeyAction::Start));
        assert_eq!(
            s.handle(VK_LCONTROL, false).action,
            Some(HotkeyAction::Stop)
        );
    }

    #[test]
    fn either_side_of_a_modifier_satisfies_the_binding() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);

        s.handle(VK_RCONTROL, true);
        assert_eq!(s.handle(VK_SPACE, true).action, Some(HotkeyAction::Toggle));
    }

    #[test]
    fn an_unrequired_modifier_blocks_the_match() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);

        s.handle(VK_LCONTROL, true);
        s.handle(VK_LSHIFT, true);
        assert_eq!(
            s.handle(VK_SPACE, true).action,
            None,
            "Ctrl+Shift+Space must not fire a Ctrl+Space binding"
        );
    }

    #[test]
    fn a_missing_modifier_blocks_the_match() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);
        assert_eq!(s.handle(VK_SPACE, true).action, None);
    }

    #[test]
    fn escape_cancels_in_hold_mode_and_is_never_swallowed() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Hold);

        let d = s.handle(VK_ESCAPE, true);
        assert_eq!(d.action, Some(HotkeyAction::Cancel));
        assert!(!d.swallow, "Escape must still reach the focused app");
    }

    /// Cancel ends the session; the trigger release that follows must stay
    /// silent rather than `Stop` a session that is already gone.
    #[test]
    fn cancel_during_a_hold_swallows_the_trailing_stop() {
        let mut s = state(vec![], "f9", ActivationMode::Hold);
        let vk_f9 = key_name_to_vk("f9").unwrap();

        assert_eq!(s.handle(vk_f9, true).action, Some(HotkeyAction::Start));
        assert_eq!(s.handle(VK_ESCAPE, true).action, Some(HotkeyAction::Cancel));
        assert_eq!(
            s.handle(vk_f9, false).action,
            None,
            "the cancelled session must not also be stopped"
        );
    }

    /// Key repeat keeps delivering key-down while the trigger stays held, so a
    /// cancel that only cleared `hold_active` would immediately restart.
    #[test]
    fn a_cancelled_hold_does_not_restart_on_key_repeat() {
        let mut s = state(vec![], "f9", ActivationMode::Hold);
        let vk_f9 = key_name_to_vk("f9").unwrap();

        s.handle(vk_f9, true);
        s.handle(VK_ESCAPE, true);
        assert_eq!(
            s.handle(vk_f9, true).action,
            None,
            "repeat must not restart"
        );

        // Letting go and pressing again is a fresh hold.
        assert_eq!(s.handle(vk_f9, false).action, None);
        assert_eq!(s.handle(vk_f9, true).action, Some(HotkeyAction::Start));
    }

    /// The mirror of the stuck-key rule the Linux backend spells out: a press
    /// the desktop received must get its release, whatever the modifiers say
    /// by then.
    #[test]
    fn a_forwarded_press_keeps_its_release() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);

        assert!(
            !s.handle(VK_SPACE, true).swallow,
            "bare Space reaches the app"
        );
        s.handle(VK_LCONTROL, true);
        assert!(
            !s.handle(VK_SPACE, false).swallow,
            "the app that got the press must get the release, or Space sticks down"
        );
    }

    /// Key repeat under a held combo must not flip the answer mid-press.
    #[test]
    fn a_repeated_swallowed_press_stays_swallowed() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);

        s.handle(VK_LCONTROL, true);
        assert!(s.handle(VK_SPACE, true).swallow);
        assert!(
            s.handle(VK_SPACE, true).swallow,
            "repeat keeps the decision"
        );
        // Letting go of the modifier first must not hand the desktop a lone
        // release for a press it never saw.
        s.handle(VK_LCONTROL, false);
        assert!(s.handle(VK_SPACE, false).swallow);
    }

    #[test]
    fn the_matched_trigger_is_swallowed_but_a_bare_press_is_not() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Toggle);

        assert!(
            !s.handle(VK_SPACE, true).swallow,
            "typing Space with no modifier must reach the app"
        );
        s.handle(VK_SPACE, false);

        s.handle(VK_LCONTROL, true);
        assert!(s.handle(VK_SPACE, true).swallow);
        assert!(
            s.handle(VK_SPACE, false).swallow,
            "the release of a swallowed press must be swallowed too"
        );
    }

    /// A hold ended by a re-target would otherwise emit a `Stop` for a combo
    /// the user never pressed.
    #[test]
    fn retarget_clears_an_active_hold_without_emitting_stop() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Hold);
        s.handle(VK_LCONTROL, true);
        assert_eq!(s.handle(VK_SPACE, true).action, Some(HotkeyAction::Start));

        s.retarget(
            WinSnapshot::from_config(&config(vec![Modifier::Alt], "f9"), ActivationMode::Hold)
                .unwrap(),
        );
        assert_eq!(s.handle(VK_SPACE, false).action, None);
    }

    #[test]
    fn switching_mode_mid_hold_does_not_double_fire() {
        let mut s = state(vec![Modifier::Ctrl], "space", ActivationMode::Hold);
        s.handle(VK_LCONTROL, true);
        s.handle(VK_SPACE, true);

        s.retarget(
            WinSnapshot::from_config(
                &config(vec![Modifier::Ctrl], "space"),
                ActivationMode::Toggle,
            )
            .unwrap(),
        );
        assert_eq!(s.handle(VK_SPACE, false).action, None);
    }

    #[test]
    fn an_unknown_toggle_key_refuses_the_whole_snapshot() {
        assert!(
            WinSnapshot::from_config(&config(vec![], "hyper"), ActivationMode::Toggle).is_none()
        );
    }

    /// An unrecognised cancel key must not take the toggle binding down with
    /// it — cancel simply stops working.
    #[test]
    fn an_unknown_cancel_key_only_disables_cancel() {
        let mut cfg = config(vec![Modifier::Ctrl], "space");
        cfg.cancel_dictation.key = "nonsense".into();
        let mut s =
            WinHotkeyState::new(WinSnapshot::from_config(&cfg, ActivationMode::Toggle).unwrap());

        assert_eq!(s.handle(VK_ESCAPE, true).action, None);
        s.handle(VK_LCONTROL, true);
        assert_eq!(s.handle(VK_SPACE, true).action, Some(HotkeyAction::Toggle));
    }
}
