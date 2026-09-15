// SPDX-License-Identifier: GPL-3.0-or-later

import { createFileRoute, lazyRouteComponent } from '@tanstack/react-router';

export const Route = createFileRoute('/dictionary')({
  component: lazyRouteComponent(
    () => import('../features/dictionary/dictionary-page'),
    'DictionaryPage',
  ),
});
