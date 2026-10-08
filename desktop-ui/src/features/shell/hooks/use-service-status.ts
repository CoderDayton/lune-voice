// SPDX-License-Identifier: GPL-3.0-or-later

import { useStore } from '@tanstack/react-store';
import { deriveServiceStatus } from '../../../lib/service-status';
import { workspaceStore } from '../../../store/workspace-store';

/** Live service status, derived once for the sidebar, the status chip and Home. */
export function useServiceStatus() {
  const status = useStore(workspaceStore, (s) => s.runtime.serviceStatus);
  return { status, ...deriveServiceStatus(status) };
}
