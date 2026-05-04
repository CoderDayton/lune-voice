import { createFileRoute, lazyRouteComponent } from '@tanstack/react-router';

export const Route = createFileRoute('/styles')({
  component: lazyRouteComponent(() => import('../features/styles/styles-page'), 'StylesPage'),
});
