
import { saveWorkspaceSetting } from "../../../store/workspace-store";
import { type SettingsProps, SettingsRow, SettingsSection } from "../components/section";
import { deliveryMethods } from "../data/settings-data";
import { useLingui } from "@lingui/react/macro";
import { TextCursorInput } from "lucide-react";

export function OutputSection({ settings }: SettingsProps) {
  const { t } = useLingui();
  return (
    <SettingsSection title={t`Text output`} description={t`Choose how finished text is inserted into the app you are currently using.`} icon={TextCursorInput}>
      <SettingsRow
        label={t`Send method`}
        description={t`Paste is simplest. Typing keystrokes can work better in some apps.`}
        control={
          <select
            className="select w-full sm:w-auto sm:min-w-56"
            value={settings.dictationMode}
            onChange={(event) => void saveWorkspaceSetting("dictationMode", event.target.value as typeof settings.dictationMode)}
          >
            {deliveryMethods.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        }
      />
    </SettingsSection>
  );
}
