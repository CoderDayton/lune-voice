// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { useEffect, useState } from 'react';
import { backendClient } from '../../../lib/backend/client';
import { SettingsRow } from './section';

/**
 * Manage the `/dev/uinput` permission that lets Lune Voice type into other apps.
 *
 * Two states, one row. When access is missing, offer to grant it — otherwise the
 * only way to get it is to trigger the lazy request by dictating, which puts a
 * password prompt in the middle of an utterance. When this app installed the
 * rule itself, offer to take it back.
 *
 * The permission is not a private preference: `TAG+="uaccess"` gives every
 * process running as the logged-in user permanent write access to
 * `/dev/uinput`, which is enough to synthesise keystrokes into any application.
 * Granting that is the price of typing text without a compositor's cooperation;
 * leaving it behind with no way to undo it is not.
 *
 * The removal half stays hidden unless this app is the one that installed it. A
 * .deb or .rpm ships the same rule and removes it on uninstall — deleting the
 * package's own file from under it would leave dpkg or rpm with a broken view of
 * itself.
 */
export function UinputRuleRow() {
  const { t } = useLingui();
  const [accessReady, setAccessReady] = useState<boolean | null>(null);
  const [selfInstalled, setSelfInstalled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([backendClient.uinputAccessReady(), backendClient.uinputRuleInstalled()])
      .then(([ready, installed]) => {
        if (cancelled) return;
        setAccessReady(ready);
        setSelfInstalled(installed);
      })
      .catch(() => {
        // A failed probe means we cannot say what the state is, so say nothing
        // rather than offering to change something we could not read.
        if (cancelled) return;
        setAccessReady(null);
        setSelfInstalled(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Still probing, or access works and this app did not arrange it — nothing to
  // offer either way.
  if (accessReady === null || (accessReady && !selfInstalled)) return null;

  const grant = async () => {
    setBusy(true);
    setError(null);
    try {
      const ready = await backendClient.grantUinputAccess();
      setAccessReady(ready);
      setSelfInstalled(await backendClient.uinputRuleInstalled().catch(() => false));
      if (!ready) {
        setError(t`Access still isn't available. A reboot may be needed for it to take effect.`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      const removed = await backendClient.removeUinputRule();
      setSelfInstalled(!removed);
      if (removed) setAccessReady(await backendClient.uinputAccessReady().catch(() => false));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <SettingsRow
        label={t`Keyboard access`}
        description={
          accessReady
            ? t`Lune Voice added a system rule so it can type into other apps and listen for your hotkey. It also lets any other program you run read and send keystrokes. Removing it stops typing and the hotkey from working until you allow it again.`
            : t`Lune Voice needs permission to type into other apps and listen for your hotkey. Allowing it also lets any other program you run read and send keystrokes. Your system will ask for your password.`
        }
        control={
          accessReady ? (
            <button
              type="button"
              className="btn btn-outline"
              onClick={() => void remove()}
              disabled={busy}
            >
              {busy ? t`Removing…` : t`Remove`}
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void grant()}
              disabled={busy}
            >
              {busy ? t`Asking…` : t`Allow`}
            </button>
          )
        }
      />
      {error ? (
        <div className="text-sm text-error px-1" role="alert">
          {error}
        </div>
      ) : null}
    </>
  );
}
