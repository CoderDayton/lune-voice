//! OS keyring–backed storage for API keys.
//!
//! API keys used to be written verbatim into `settings.toml`. This module keeps
//! them in the platform credential store instead (Secret Service on Linux,
//! Credential Manager on Windows, Keychain on macOS) so they are encrypted at
//! rest and never touch the plaintext config file.
//!
//! Degradation is deliberate and non-fatal: when no keyring is available
//! (headless Linux without a Secret Service, a locked session, CI) reads return
//! `None` and writes report failure, so the caller keeps the key inline in the
//! TOML exactly as before. A key is therefore never lost — security is
//! best-effort, correctness is guaranteed.

/// Service name registered under the OS credential store. Shared by every slot;
/// the [`SecretSlot`] account name disambiguates the individual secrets.
const SERVICE: &str = "lune-voice";

/// Logical secret slots persisted in the keyring. The variant maps to a stable
/// account name — renaming these strands existing credentials, so don't.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecretSlot {
    /// Remote OpenAI-compatible **ASR** provider key (`asr_backend`).
    AsrApiKey,
    /// Remote OpenAI-compatible **styling-LLM** provider key (`llm_openai`).
    LlmApiKey,
}

impl SecretSlot {
    /// Stable keyring account name. Part of the on-disk credential identity.
    fn account(self) -> &'static str {
        match self {
            SecretSlot::AsrApiKey => "asr_api_key",
            SecretSlot::LlmApiKey => "llm_api_key",
        }
    }
}

/// A store for secret strings.
///
/// Abstracted behind a trait so tests inject an in-memory backend instead of
/// mutating the developer's real OS keyring.
pub trait SecretVault: Send + Sync {
    /// Fetch a secret. Returns `None` when the slot is empty **or** the vault is
    /// unavailable — callers treat both identically and fall back to any inline
    /// value from the TOML.
    fn get(&self, slot: SecretSlot) -> Option<String>;

    /// Store a secret. Returns `true` only when it was durably written to the
    /// secure store; `false` means the caller must keep the key inline as a
    /// fallback so it is not lost.
    fn set(&self, slot: SecretSlot, value: &str) -> bool;

    /// Remove a secret. Best-effort — a missing entry or an unavailable vault is
    /// not an error.
    fn delete(&self, slot: SecretSlot);
}

/// [`SecretVault`] backed by the platform credential store via the `keyring`
/// crate. Every operation swallows keyring errors into the graceful-degradation
/// contract documented on the trait.
pub struct KeyringVault;

impl KeyringVault {
    fn entry(slot: SecretSlot) -> keyring::Result<keyring::Entry> {
        keyring::Entry::new(SERVICE, slot.account())
    }
}

impl SecretVault for KeyringVault {
    fn get(&self, slot: SecretSlot) -> Option<String> {
        match Self::entry(slot).and_then(|e| e.get_password()) {
            Ok(value) => Some(value),
            Err(keyring::Error::NoEntry) => None,
            Err(err) => {
                tracing::debug!(
                    slot = slot.account(),
                    error = %err,
                    "keyring read unavailable; treating API key as absent",
                );
                None
            }
        }
    }

    fn set(&self, slot: SecretSlot, value: &str) -> bool {
        match Self::entry(slot).and_then(|e| e.set_password(value)) {
            Ok(()) => true,
            Err(err) => {
                tracing::warn!(
                    slot = slot.account(),
                    error = %err,
                    "keyring write failed; API key will remain in settings.toml",
                );
                false
            }
        }
    }

    fn delete(&self, slot: SecretSlot) {
        let Ok(entry) = Self::entry(slot) else {
            return;
        };
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(err) => tracing::debug!(
                slot = slot.account(),
                error = %err,
                "keyring delete failed",
            ),
        }
    }
}

/// In-memory [`SecretVault`] for tests — never touches the real OS keyring.
#[cfg(test)]
#[derive(Default)]
pub struct InMemoryVault {
    inner: std::sync::Mutex<std::collections::HashMap<&'static str, String>>,
}

#[cfg(test)]
impl SecretVault for InMemoryVault {
    fn get(&self, slot: SecretSlot) -> Option<String> {
        self.inner.lock().unwrap().get(slot.account()).cloned()
    }

    fn set(&self, slot: SecretSlot, value: &str) -> bool {
        self.inner
            .lock()
            .unwrap()
            .insert(slot.account(), value.to_owned());
        true
    }

    fn delete(&self, slot: SecretSlot) {
        self.inner.lock().unwrap().remove(slot.account());
    }
}
