// SPDX-License-Identifier: GPL-3.0-or-later

import { useLingui } from '@lingui/react/macro';
import { HardDriveDownload } from 'lucide-react';
import { ModelLibrary } from '../components/model-library';
import { SettingsSection } from '../components/section';
import type { ModelStatusApi } from '../hooks/use-model-status';

export function ModelsSection({ status }: { status: ModelStatusApi }) {
  const { t } = useLingui();
  return (
    <SettingsSection
      title={t`Models`}
      description={t`Download and delete the speech models stored on this machine.`}
      icon={HardDriveDownload}
    >
      <ModelLibrary status={status} />
    </SettingsSection>
  );
}
