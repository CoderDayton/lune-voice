import { useLingui } from '@lingui/react/macro';
import { useEffect, useState } from 'react';
import { backendClient } from '../../../lib/backend/client';
import { SettingsRow } from './section';

/**
 * Offer to remove the `/dev/uinput` udev rule this app installed at first run.
 *
 * The rule is not a private preference. `TAG+="uaccess"` gives every process
 * running as the logged-in user permanent write access to `/dev/uinput`, which
 * is enough to synthesise keystrokes into any application. Granting that is the
 * price of typing text without a compositor's cooperation; leaving it behind
 * with no way to undo it is not.
 *
 * Renders nothing unless this app is the one that installed it. A .deb or .rpm
 * ships the same rule and removes it on uninstall — deleting the package's own
 * file from under it would leave dpkg or rpm with a broken view of itself.
 */
export function UinputRuleRow() {
  const { t } = useLingui();
  const [installed, setInstalled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void backendClient
      .uinputRuleInstalled()
      .then((present) => {
        if (!cancelled) setInstalled(present);
      })
      .catch(() => {
        // A failed probe means we cannot say the rule is there, so say nothing.
        if (!cancelled) setInstalled(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!installed) return null;

  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      const removed = await backendClient.removeUinputRule();
      setInstalled(!removed);
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
        description={t`Lune Voice added a system rule so it can type into other apps. It also lets any other program you run send keystrokes. Removing it stops typed delivery from working until you allow it again.`}
        control={
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => void remove()}
            disabled={busy}
          >
            {busy ? t`Removing…` : t`Remove`}
          </button>
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
