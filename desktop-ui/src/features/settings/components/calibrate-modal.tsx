import { useLingui } from '@lingui/react/macro';
import { useEffect, useId, useState } from 'react';
import { AudioDots } from '../../dictation/audio-dots';
import { useCalibration } from '../hooks/use-calibration';

interface CalibrateModalProps {
  open: boolean;
  onClose: () => void;
  currentStart: number;
  currentEnd: number;
}

export function CalibrateModal({ open, onClose, currentStart, currentEnd }: CalibrateModalProps) {
  const { t } = useLingui();
  const titleId = useId();
  const [dialog, setDialog] = useState<HTMLDialogElement | null>(null);
  const cal = useCalibration();
  const promptSentence = t`Please call Stella. Ask her to bring these things with her from the store: six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.`;

  useEffect(() => {
    if (!dialog) return;
    if (open && !dialog.open) {
      void cal.start();
      dialog.showModal();
    }
    if (!open && dialog.open) dialog.close();
  }, [open, dialog, cal]);

  useEffect(() => {
    if (!dialog) return;
    // The browser fires `close` after either Esc (preceded by `cancel`)
    // or close()/form-dialog submission, so listening on `close` alone
    // covers every dismissal path.
    const onClose_ = () => {
      void cal.cancel();
      onClose();
    };
    dialog.addEventListener('close', onClose_);
    return () => {
      dialog.removeEventListener('close', onClose_);
    };
  }, [dialog, cal, onClose]);

  const stepNum = stepFor(cal.phase);
  const totalSteps = 3;

  return (
    <dialog
      ref={(el) => {
        setDialog(el);
        return () => setDialog(null);
      }}
      className="modal"
      aria-labelledby={titleId}
    >
      <div className="modal-box w-full max-w-md sm:max-w-lg border border-surface-border">
        <div className="flex items-center justify-between mb-3">
          <h3 id={titleId} className="text-xl font-bold">
            {t`Calibrate microphone`}
          </h3>
          <span className="text-xs text-content-muted tabular-nums">
            {stepNum} / {totalSteps}
          </span>
        </div>

        {cal.phase === 'noise' && (
          <PhaseNoise
            level={cal.energy}
            description={t`Stay quiet — measuring background noise.`}
          />
        )}

        {cal.phase === 'waiting' && (
          <PhaseWaiting
            prompt={promptSentence}
            onStart={() => void cal.advance()}
            startLabel={t`Start reading`}
            description={t`When you're ready, click Start and read the sentence aloud at your normal pace.`}
          />
        )}

        {cal.phase === 'speech' && (
          <PhaseSpeech
            prompt={promptSentence}
            level={cal.energy}
            description={t`Read aloud. Don't worry about pace — just speak naturally.`}
          />
        )}

        {cal.phase === 'done' && cal.recommendation && (
          <PhaseResult
            recommended={cal.recommendation}
            currentStart={currentStart}
            currentEnd={currentEnd}
            applyLabel={t`Apply`}
            adjustLabel={t`Adjust manually`}
            recalibrateLabel={t`Recalibrate`}
            startLabel={t`Start sensitivity`}
            pauseLabel={t`Pause sensitivity`}
            currentLabel={t`Current`}
            recommendedLabel={t`Recommended`}
            onApply={async () => {
              await cal.apply();
              onClose();
            }}
            onAdjust={onClose}
            onRecalibrate={() => void cal.start()}
          />
        )}

        {cal.phase === 'failed' && (
          <PhaseError
            message={cal.errorMessage ?? t`Something went wrong.`}
            retryLabel={t`Retry`}
            cancelLabel={t`Cancel`}
            onRetry={() => void cal.start()}
            onCancel={onClose}
          />
        )}

        {cal.phase === 'idle' && <p className="py-4 text-sm text-content-muted">{t`Starting…`}</p>}
      </div>
      <form method="dialog" className="modal-backdrop">
        <button type="submit">close</button>
      </form>
    </dialog>
  );
}

function stepFor(phase: string): number {
  switch (phase) {
    case 'noise':
      return 1;
    case 'waiting':
    case 'speech':
      return 2;
    case 'done':
    case 'failed':
      return 3;
    default:
      return 1;
  }
}

interface PhaseNoiseProps {
  level: number;
  description: string;
}

function PhaseNoise({ level, description }: PhaseNoiseProps) {
  return (
    <div className="py-4 flex flex-col gap-3">
      <p className="text-sm text-content-muted">{description}</p>
      <div className="flex justify-center py-2">
        <AudioDots level={level} />
      </div>
    </div>
  );
}

interface PhaseWaitingProps {
  prompt: string;
  description: string;
  onStart: () => void;
  startLabel: string;
}

function PhaseWaiting({ prompt, description, onStart, startLabel }: PhaseWaitingProps) {
  return (
    <div className="py-4 flex flex-col gap-3">
      <p className="text-sm text-content-muted">{description}</p>
      <p className="text-base leading-relaxed bg-surface-subtle rounded-field p-3 border border-surface-border">
        {prompt}
      </p>
      <div className="modal-action">
        <button type="button" className="btn btn-primary" onClick={onStart}>
          {startLabel}
        </button>
      </div>
    </div>
  );
}

interface PhaseSpeechProps {
  prompt: string;
  level: number;
  description: string;
}

function PhaseSpeech({ prompt, level, description }: PhaseSpeechProps) {
  return (
    <div className="py-4 flex flex-col gap-3">
      <p className="text-sm text-content-muted">{description}</p>
      <p className="text-base leading-relaxed bg-surface-subtle rounded-field p-3 border border-surface-border">
        {prompt}
      </p>
      <div className="flex justify-center py-1">
        <AudioDots level={level} />
      </div>
    </div>
  );
}

interface PhaseResultProps {
  recommended: { speechThreshold: number; speechThresholdEnd: number };
  currentStart: number;
  currentEnd: number;
  startLabel: string;
  pauseLabel: string;
  currentLabel: string;
  recommendedLabel: string;
  applyLabel: string;
  adjustLabel: string;
  recalibrateLabel: string;
  onApply: () => void | Promise<void>;
  onAdjust: () => void;
  onRecalibrate: () => void;
}

function PhaseResult(props: PhaseResultProps) {
  const fmt = (n: number) => n.toFixed(2);
  return (
    <div className="py-4 flex flex-col gap-3">
      <div className="grid grid-cols-3 gap-2 text-sm sm:gap-3">
        <div />
        <div className="text-content-muted text-xs uppercase tracking-wide">
          {props.currentLabel}
        </div>
        <div className="text-content-muted text-xs uppercase tracking-wide">
          {props.recommendedLabel}
        </div>
        <div className="font-medium">{props.startLabel}</div>
        <div className="tabular-nums">{fmt(props.currentStart)}</div>
        <div className="tabular-nums font-semibold text-primary">
          {fmt(props.recommended.speechThreshold)}
        </div>
        <div className="font-medium">{props.pauseLabel}</div>
        <div className="tabular-nums">{fmt(props.currentEnd)}</div>
        <div className="tabular-nums font-semibold text-primary">
          {fmt(props.recommended.speechThresholdEnd)}
        </div>
      </div>
      <div className="modal-action gap-2">
        <button type="button" className="btn btn-soft" onClick={props.onRecalibrate}>
          {props.recalibrateLabel}
        </button>
        <button type="button" className="btn btn-soft" onClick={props.onAdjust}>
          {props.adjustLabel}
        </button>
        <button type="button" className="btn btn-primary" onClick={() => void props.onApply()}>
          {props.applyLabel}
        </button>
      </div>
    </div>
  );
}

interface PhaseErrorProps {
  message: string;
  retryLabel: string;
  cancelLabel: string;
  onRetry: () => void;
  onCancel: () => void;
}

function PhaseError({ message, retryLabel, cancelLabel, onRetry, onCancel }: PhaseErrorProps) {
  return (
    <div className="py-4 flex flex-col gap-3">
      <p className="text-sm text-error">{message}</p>
      <div className="modal-action gap-2">
        <button type="button" className="btn btn-soft" onClick={onCancel}>
          {cancelLabel}
        </button>
        <button type="button" className="btn btn-primary" onClick={onRetry}>
          {retryLabel}
        </button>
      </div>
    </div>
  );
}
