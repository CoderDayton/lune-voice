import { useLingui } from '@lingui/react/macro';
import { Check, CircleAlert, Loader2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { backendClient } from '../../../lib/backend/client';
import { workspaceStore } from '../../../store/workspace-store';

type Probe<T> = { state: 'checking' } | { state: 'ok'; value: T } | { state: 'failed' };

/**
 * The two things setup could not previously tell the user about, asked before
 * they leave the screen rather than during their first dictation.
 *
 * Both are permission-or-hardware questions that a download step cannot answer:
 * a machine with no input device finishes every step and then transcribes
 * silence, and keyboard access is otherwise requested by a password prompt that
 * appears mid-sentence — where it reads as a malfunction, and where cancelling
 * it loses the words already spoken.
 */
export function ReadyChecks() {
  const { t } = useLingui();
  const [mic, setMic] = useState<Probe<string[]>>({ state: 'checking' });
  const [keyboard, setKeyboard] = useState<Probe<boolean>>({ state: 'checking' });
  const [granting, setGranting] = useState(false);
  const [grantError, setGrantError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void backendClient
      .listAudioInputDevices()
      .then((devices) => {
        if (!cancelled) setMic({ state: 'ok', value: devices });
      })
      .catch(() => {
        if (!cancelled) setMic({ state: 'failed' });
      });
    void backendClient
      .uinputAccessReady()
      .then((ready) => {
        if (!cancelled) setKeyboard({ state: 'ok', value: ready });
      })
      .catch(() => {
        if (!cancelled) setKeyboard({ state: 'failed' });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const grant = useCallback(async () => {
    setGranting(true);
    setGrantError(null);
    try {
      const ready = await backendClient.grantUinputAccess();
      setKeyboard({ state: 'ok', value: ready });
      if (!ready) {
        setGrantError(
          t`Access still isn't available. A reboot may be needed for it to take effect.`,
        );
      }
    } catch (e) {
      // The backend messages name the actual step that failed ("authentication
      // cancelled", "could not write the udev rule"), which is more use than a
      // generic refusal.
      setGrantError(e instanceof Error ? e.message : String(e));
    } finally {
      setGranting(false);
    }
  }, [t]);

  const micDevices = mic.state === 'ok' ? mic.value : [];
  const micOk = mic.state === 'ok' && micDevices.length > 0;
  // What dictation will actually record from, not whatever happens to be first
  // in the enumeration. `null` is the setting's "system default", and a saved
  // device that is no longer plugged in falls back to the default too.
  const savedDevice = workspaceStore.state.settings.inputDevice;
  const activeDevice =
    savedDevice && micDevices.includes(savedDevice) ? savedDevice : t`your system default mic`;
  const keyboardOk = keyboard.state === 'ok' && keyboard.value;

  return (
    <section>
      {/* The caption sits outside the box. Inside, it was a header bar — a
          container style used nowhere else on either screen, which made this
          block read as a different kind of thing from the cards above it. */}
      <span className="setup-label">{t`Before you start`}</span>
      <div className="setup-group">
        <CheckRow
          label={t`Microphone`}
          pending={mic.state === 'checking'}
          ok={micOk}
          detail={
            mic.state === 'checking'
              ? t`Looking for input devices…`
              : micOk
                ? // Naming the device is the difference between "a mic exists"
                  // and "the right mic is picked" — the second is what actually
                  // goes wrong on a machine with a webcam and a headset.
                  t`Using ${activeDevice}. Change it in Settings › Dictation.`
                : mic.state === 'failed'
                  ? t`Couldn't check your input devices.`
                  : t`No microphone found. Plug one in, then pick it in Settings › Dictation.`
          }
        />

        <CheckRow
          label={t`Keyboard access`}
          pending={keyboard.state === 'checking' || granting}
          ok={keyboardOk}
          detail={
            keyboard.state === 'checking'
              ? t`Checking…`
              : keyboardOk
                ? t`Lune Voice can type into your other apps.`
                : t`Lune Voice needs permission to type into other apps. Your system will ask for your password.`
          }
          action={
            keyboard.state !== 'checking' && !keyboardOk ? (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={granting}
                onClick={() => void grant()}
              >
                {granting ? t`Asking…` : t`Allow`}
              </button>
            ) : null
          }
        />

        {grantError && (
          <p
            role="alert"
            className="border-t border-surface-border px-4 py-3 text-xs break-words text-error"
          >
            {grantError}
          </p>
        )}
      </div>
    </section>
  );
}

interface CheckRowProps {
  label: string;
  pending: boolean;
  ok: boolean;
  detail: string;
  action?: React.ReactNode;
}

function CheckRow({ label, pending, ok, detail, action }: CheckRowProps) {
  const tone = pending
    ? 'bg-base-300 text-content-subtle'
    : ok
      ? 'bg-success/15 text-success'
      : 'bg-warning/15 text-warning';

  return (
    <div className="setup-row-divider flex items-start gap-3 p-4">
      {/* The tile carries the status, not the subject. It used to show a
          green check while a second, smaller mic glyph sat next to the word
          "Microphone" — two icons for one row, and the label already says
          which check this is. */}
      <span className={`setup-tile ${tone}`} aria-hidden="true">
        {pending ? (
          <Loader2 size={17} className="animate-spin" />
        ) : ok ? (
          <Check size={17} strokeWidth={2.5} />
        ) : (
          <CircleAlert size={17} strokeWidth={1.9} />
        )}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-content-strong">{label}</p>
        <p className="mt-0.5 text-xs leading-snug text-content-subtle">{detail}</p>
      </div>
      {action && <div className="shrink-0 self-center">{action}</div>}
    </div>
  );
}
