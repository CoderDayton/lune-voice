import { useLingui } from '@lingui/react/macro';
import { Check, Cloud, Cpu, type LucideIcon, PowerOff, Wand2 } from 'lucide-react';
import { saveWorkspaceSettings } from '../../../store/workspace-store';
import { OpenAiFields } from '../components/openai-fields';
import { SettingsBlock, type SettingsProps, SettingsSection } from '../components/section';
import { type LlmEngineValue, llmEngines } from '../data/settings-data';

function engineIcon(value: LlmEngineValue): LucideIcon {
  if (value === 'OpenAiCompatible') return Cloud;
  if (value === 'Local') return Cpu;
  return PowerOff;
}

/**
 * Styling-engine picker: Off / On-device / OpenAI-compatible (remote). This is
 * the single source of truth for the styling LLM backend — selecting an engine
 * enables styling with it; Off disables styling entirely. The Styles tab toggle
 * is a shortcut that maps onto the same settings.
 */
export function StylingEngineSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  const current: LlmEngineValue = !settings.llmEnabled ? 'Disabled' : settings.llmBackend;

  function selectEngine(value: LlmEngineValue) {
    if (value === 'Disabled') {
      // Turning styling off also drops the selected preset — otherwise the
      // settings migration re-heals it back on the next load.
      void saveWorkspaceSettings({ llmEnabled: false, stylePreset: null });
    } else {
      void saveWorkspaceSettings({ llmEnabled: true, llmBackend: value });
    }
  }

  return (
    <div className="flex flex-col gap-space-md">
      <SettingsSection
        title={t`Styling engine`}
        description={t`Choose how transcripts are restyled. On-device runs a local model; the cloud option sends transcripts to a remote OpenAI-compatible endpoint you configure.`}
        icon={Wand2}
      >
        <SettingsBlock
          label={t`Engine`}
          description={t`Off delivers raw transcripts. Which style is applied is chosen on the Styles tab.`}
        >
          <div className="grid gap-2 sm:grid-cols-3">
            {llmEngines.map((engine) => (
              <EngineCard
                key={engine.value}
                label={engine.label}
                description={engine.description ?? ''}
                icon={engineIcon(engine.value)}
                active={current === engine.value}
                onSelect={() => selectEngine(engine.value)}
              />
            ))}
          </div>
        </SettingsBlock>
        {current === 'OpenAiCompatible' && (
          <OpenAiFields
            settings={settings}
            baseUrlKey="openaiLlmBaseUrl"
            apiKeyKey="openaiLlmApiKey"
            modelKey="openaiLlmModel"
            apiDescription="styling API"
            modelDescription="Styling model name, e.g. gpt-4o-mini."
            modelPlaceholder="gpt-4o-mini"
          />
        )}
      </SettingsSection>
    </div>
  );
}

interface EngineCardProps {
  label: string;
  description: string;
  icon: LucideIcon;
  active: boolean;
  onSelect: () => void;
}

function EngineCard({ label, description, icon: Icon, active, onSelect }: EngineCardProps) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={`group flex flex-col gap-2 rounded-box border p-3 text-left transition-all duration-fast ${
        active
          ? 'border-primary/50 bg-primary/8 shadow-sm'
          : 'border-surface-border bg-surface-inset hover:border-primary/40 hover:bg-primary/5'
      }`}
    >
      <div className="flex items-center justify-between">
        <span
          className={`flex size-9 items-center justify-center rounded-field transition-colors ${
            active
              ? 'bg-primary/15 text-primary'
              : 'bg-surface-elevated text-content-subtle group-hover:text-content'
          }`}
        >
          <Icon size={17} strokeWidth={1.7} />
        </span>
        {active && <Check size={16} strokeWidth={2.5} className="text-primary" />}
      </div>
      <div>
        <div className="font-medium text-content-strong text-sm">{label}</div>
        <div className="mt-0.5 text-content-muted text-xs">{description}</div>
      </div>
    </button>
  );
}
