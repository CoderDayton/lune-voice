// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { SidebarQuickStart } from '../../src/features/shell/components/sidebar-widgets';
import { SIDEBAR_NAV_ID } from '../../src/features/shell/config/drawer';
import { resetMockBackend } from '../../src/lib/backend/mock';
import { workspaceStore } from '../../src/store/workspace-store';

i18n.load('en', {});
i18n.activate('en');

const initialStatus = workspaceStore.state.runtime.serviceStatus;

beforeAll(() => {
  GlobalRegistrator.register();
});

beforeEach(() => {
  resetMockBackend();
  // The button is disabled while the service is still connecting.
  workspaceStore.setState((prev) => ({
    ...prev,
    runtime: { ...prev.runtime, serviceStatus: 'ready' },
  }));
});

afterEach(() => {
  cleanup();
});

afterAll(async () => {
  cleanup();
  workspaceStore.setState((prev) => ({
    ...prev,
    runtime: { ...prev.runtime, serviceStatus: initialStatus },
  }));
  // Let React finish its last scheduled render before the DOM globals go.
  await new Promise((resolve) => setTimeout(resolve, 50));
  await GlobalRegistrator.unregister();
});

function renderQuickStart() {
  const view = render(
    <I18nProvider i18n={i18n}>
      <SidebarQuickStart />
      <nav id={SIDEBAR_NAV_ID} tabIndex={-1} />
    </I18nProvider>,
  );
  const button = view.getByRole('button', { name: 'Start dictating' }) as HTMLButtonElement;
  return { ...view, button };
}

describe('SidebarQuickStart focus after toggling dictation', () => {
  test('a keyboard activation parks focus on the sidebar nav', () => {
    const { button } = renderQuickStart();

    button.focus();
    fireEvent.keyDown(button, { key: 'Enter' });
    fireEvent.click(button);

    expect(document.activeElement?.id).toBe(SIDEBAR_NAV_ID);
  });

  test('a mouse press leaves focus off the nav', () => {
    const { button } = renderQuickStart();

    fireEvent.pointerDown(button);
    fireEvent.click(button);

    expect(document.activeElement).toBe(document.body);
  });

  test('a click with no key or pointer press, such as a script or assistive tech, leaves focus alone', () => {
    const { button } = renderQuickStart();

    fireEvent.click(button);

    expect(document.activeElement).toBe(document.body);
  });

  test('a key press does not carry over to a later mouse click', () => {
    const { button } = renderQuickStart();

    fireEvent.keyDown(button, { key: 'Enter' });
    fireEvent.pointerDown(button);
    fireEvent.click(button);

    expect(document.activeElement).toBe(document.body);
  });
});
