// SPDX-License-Identifier: GPL-3.0-or-later

import { createFileRoute, redirect } from '@tanstack/react-router';

export const Route = createFileRoute('/shortcuts')({
  beforeLoad: () => {
    throw redirect({ to: '/settings' });
  },
});
