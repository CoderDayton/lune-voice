// SPDX-License-Identifier: GPL-3.0-or-later

import { afterAll, afterEach, beforeAll, describe, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { RailTooltip } from '../../src/features/shell/components/rail-tooltip';

beforeAll(() => {
  GlobalRegistrator.register();
});

afterEach(() => {
  cleanup();
});

afterAll(async () => {
  cleanup();
  // React schedules its last render on a timer. Let it run before the DOM
  // globals go, or it throws `window is not defined` into the next file.
  await new Promise((resolve) => setTimeout(resolve, 50));
  await GlobalRegistrator.unregister();
});

/** The drawer checkbox sets whether the sidebar counts as expanded. */
function renderRail(expanded: boolean) {
  const view = render(
    <>
      <input id="app-drawer" type="checkbox" defaultChecked={expanded} />
      <RailTooltip label="Settings" hint="Ctrl+,">
        <button type="button">gear</button>
      </RailTooltip>
    </>,
  );
  const button = view.getByRole('button') as HTMLButtonElement;
  const wrapper = button.parentElement as HTMLElement;
  return { ...view, button, wrapper };
}

function rect(right: number): DOMRect {
  return {
    x: 0,
    y: 40,
    left: 0,
    top: 40,
    right,
    bottom: 60,
    width: 20,
    height: 20,
    toJSON() {
      return this;
    },
  } as DOMRect;
}

/** Long enough to pass the 300 ms pointer delay. */
function afterShowDelay() {
  return new Promise((resolve) => setTimeout(resolve, 400));
}

describe('RailTooltip', () => {
  test('shows its label after the pointer rests on a collapsed control', async () => {
    const { wrapper, findByText } = renderRail(false);

    fireEvent.pointerEnter(wrapper);

    expect(await findByText('Settings')).toBeDefined();
  });

  test('stays hidden while the sidebar is expanded, because the label is on screen', async () => {
    const { wrapper, queryByText } = renderRail(true);

    fireEvent.pointerEnter(wrapper);
    await afterShowDelay();

    expect(queryByText('Settings')).toBeNull();
  });

  test('shows at once when the control takes keyboard focus', async () => {
    const { button, wrapper, findByText } = renderRail(false);

    button.focus();
    fireEvent.focusIn(wrapper);

    expect(await findByText('Settings')).toBeDefined();
  });

  test('keeps the label when the pointer leaves a control that still has focus', async () => {
    const { button, wrapper, findByText, queryByText } = renderRail(false);

    button.focus();
    fireEvent.focusIn(wrapper);
    await findByText('Settings');
    fireEvent.pointerLeave(wrapper);

    expect(document.activeElement).toBe(button);
    expect(queryByText('Settings')).not.toBeNull();
  });

  test('hides the label when the pointer leaves a control without focus', async () => {
    const { wrapper, findByText, queryByText } = renderRail(false);

    fireEvent.pointerEnter(wrapper);
    await findByText('Settings');
    fireEvent.pointerLeave(wrapper);

    expect(queryByText('Settings')).toBeNull();
  });

  test('follows its control when a scroll container moves it', async () => {
    const { button, wrapper, findByText, getByText } = renderRail(false);
    button.getBoundingClientRect = () => rect(100);

    fireEvent.pointerEnter(wrapper);
    await findByText('Settings');
    expect(getByText('Settings').parentElement?.style.left).toBe('108px');

    button.getBoundingClientRect = () => rect(300);
    act(() => {
      window.dispatchEvent(new Event('scroll'));
    });

    await waitFor(() => {
      expect(getByText('Settings').parentElement?.style.left).toBe('308px');
    });
  });
});
