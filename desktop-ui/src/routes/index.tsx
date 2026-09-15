// SPDX-License-Identifier: GPL-3.0-or-later

import { createFileRoute, lazyRouteComponent } from '@tanstack/react-router';

export const Route = createFileRoute('/')({
  component: lazyRouteComponent(() => import('../features/overview/overview-page'), 'OverviewPage'),
});
