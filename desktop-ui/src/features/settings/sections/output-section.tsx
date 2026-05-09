import { useLingui } from '@lingui/react/macro';
import { TextCursorInput } from 'lucide-react';
import { DescriptiveDropdown } from '../../../components/ui/descriptive-dropdown';
import { saveWorkspaceSetting } from '../../../store/workspace-store';
import { type SettingsProps, SettingsRow, SettingsSection } from '../components/section';
import { deliveryMethods } from '../data/settings-data';

export function OutputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  return (
    <SettingsSection
      title={t`Text output`}
      description={t`Choose how finished text is inserted into the app you are currently using.`}
      icon={TextCursorInput}
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
    </SettingsSection>
  );
}
