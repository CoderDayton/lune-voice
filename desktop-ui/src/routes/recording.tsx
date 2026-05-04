import { createFileRoute, redirect } from '@tanstack/react-router';

export const Route = createFileRoute('/recording')({
  beforeLoad: () => {
    throw redirect({ to: '/settings' });
  },
});
