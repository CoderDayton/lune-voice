// SPDX-License-Identifier: GPL-3.0-or-later

import { createFileRoute, lazyRouteComponent } from '@tanstack/react-router';

export const Route = createFileRoute('/history')({
  component: lazyRouteComponent(() => import('../features/history/history-page'), 'HistoryPage'),
});
