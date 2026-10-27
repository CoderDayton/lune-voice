// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { i18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { cleanup, render } from '@testing-library/react';
import { Home } from 'lucide-react';
import { ActionMenu } from '../../src/components/ui/action-menu';
import { NavBead } from '../../src/components/ui/nav-item';
import { SidebarQuickStart } from '../../src/features/shell/components/sidebar-widgets';
import { resetMockBackend } from '../../src/lib/backend/mock';
import { workspaceStore } from '../../src/store/workspace-store';

i18n.load('en', {});
i18n.activate('en');

const initialStatus = workspaceStore.state.runtime.serviceStatus;

function setStatus(serviceStatus: string) {
  workspaceStore.setState((prev) => ({
    ...prev,
    runtime: { ...prev.runtime, serviceStatus },
  }));
}

beforeAll(() => {
  GlobalRegistrator.register();
});

beforeEach(() => {
  resetMockBackend();
  setStatus('ready');
});

afterEach(() => {
  cleanup();
});

afterAll(async () => {
  cleanup();
  setStatus(initialStatus);
  // Let React finish its last scheduled render before the DOM globals go.
  await new Promise((resolve) => setTimeout(resolve, 50));
  await GlobalRegistrator.unregister();
});

describe('NavBead', () => {
  test('lights up only when active, and hides from assistive tech', () => {
    const off = render(
      <NavBead active={false}>
        <Home size={16} />
      </NavBead>,
    ).container.firstElementChild as HTMLElement;
    cleanup();
    const on = render(
      <NavBead active>
        <Home size={16} />
      </NavBead>,
    ).container.firstElementChild as HTMLElement;

    expect(off.getAttribute('aria-hidden')).toBe('true');
    expect(on.getAttribute('aria-hidden')).toBe('true');
    expect(off.className).toContain('rounded-full');
    expect(on.className).toContain('rounded-full');
    expect(on.className).toContain('bg-primary/20');
    expect(off.className).not.toContain('bg-primary/20');
  });
});

describe('SidebarQuickStart look', () => {
  function renderButton() {
    const view = render(
      <I18nProvider i18n={i18n}>
        <SidebarQuickStart />
      </I18nProvider>,
    );
    return view.getByRole('button') as HTMLButtonElement;
  }

  test('rests as a brand-gradient moon with no ring', () => {
    const button = renderButton();
    expect(button.innerHTML).toContain('surface-brand');
    expect(button.querySelector('.animate-pulse-ring')).toBeNull();
  });

  test('pulses a ring while a session is live', () => {
    setStatus('listening');
    const button = renderButton();
    expect(button.querySelector('.animate-pulse-ring')).not.toBeNull();
    expect(button.innerHTML).not.toContain('surface-brand');
  });
});

describe('ActionMenu look', () => {
  test('puts each icon in a round chip and eases the menu in', () => {
    const anchor = document.createElement('button');
    document.body.appendChild(anchor);
    render(
      <ActionMenu
        ariaLabel="More"
        open
        onOpenChange={() => {}}
        anchorRef={{ current: anchor }}
        items={[
          { id: 'a', label: 'Alpha', icon: Home, trailing: 'Ctrl+K', onSelect: () => {} },
          { id: 'b', label: 'Beta', icon: Home, danger: true, onSelect: () => {} },
        ]}
      />,
    );

    const menu = document.body.querySelector('[role="menu"]') as HTMLElement;
    expect(menu.className).toContain('animate-page-in');
    const rows = menu.querySelectorAll('[role="menuitem"]');
    expect(rows.length).toBe(2);
    for (const row of rows) {
      expect(row.querySelector('span.rounded-full svg')).not.toBeNull();
    }
    expect(rows[0]?.querySelector('kbd')?.textContent).toBe('Ctrl+K');
  });
});
