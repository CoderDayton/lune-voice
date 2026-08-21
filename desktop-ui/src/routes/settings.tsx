import { createFileRoute, lazyRouteComponent } from '@tanstack/react-router';
import { isSettingsSectionId, type SettingsSectionId } from '../features/settings/sections';

export const Route = createFileRoute('/settings')({
  // `?tab=` lets other surfaces deep-link a section — the sidebar's Quick
  // actions menu opens Dictation this way. An unknown value falls back to the
  // page default rather than erroring.
  validateSearch: (search: Record<string, unknown>): { tab?: SettingsSectionId } =>
    isSettingsSectionId(search.tab) ? { tab: search.tab } : {},
  component: lazyRouteComponent(() => import('../features/settings/settings-page'), 'SettingsPage'),
});
