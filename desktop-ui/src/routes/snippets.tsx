// SPDX-License-Identifier: GPL-3.0-or-later

import { createFileRoute, lazyRouteComponent } from '@tanstack/react-router';

export const Route = createFileRoute('/snippets')({
  component: lazyRouteComponent(() => import('../features/snippets/snippets-page'), 'SnippetsPage'),
});
