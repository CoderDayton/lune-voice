import { useLingui } from '@lingui/react/macro';
import { Send } from 'lucide-react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { deliveryMethods, terminalModes } from '../data/settings-data';

export function OutputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  return (
    <SettingsSection
      title={t`Output`}
      description={t`Choose how finished text is inserted into the app you are currently using.`}
      icon={Send}
    >
      <SettingsRow
        label={t`Send method`}
        description={t`Paste is simplest. Typing keystrokes can work better in some apps.`}
        control={
          <DescriptiveDropdown
            value={settings.dictationMode}
            options={deliveryMethods}
            ariaLabel={t`Send method`}
            onChange={(next) =>
              void saveWorkspaceSetting('dictationMode', next as typeof settings.dictationMode)
            }
          />
        }
      />
      <SettingsRow
        label={t`Copy to clipboard`}
        description={t`Keep the dictated text on the clipboard after every dictation, so you can paste it again yourself.`}
        control={
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={settings.copyToClipboard}
            onChange={() => void saveWorkspaceSetting('copyToClipboard', !settings.copyToClipboard)}
            aria-label={t`Copy to clipboard`}
          />
        }
      />
      {settings.dictationMode !== 'TypeInPlace' && (
        <SettingsRow
          label={t`Paste compatibility`}
          description={t`Which paste shortcut to send. Automatic detects terminals; override if paste lands in the wrong place.`}
          control={
            <DescriptiveDropdown
              value={settings.terminalMode}
              options={terminalModes}
              ariaLabel={t`Paste compatibility`}
              onChange={(next) =>
                void saveWorkspaceSetting('terminalMode', next as typeof settings.terminalMode)
              }
            />
          }
        />
      )}
    </SettingsSection>
  );
}
