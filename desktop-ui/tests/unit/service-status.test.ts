// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from "bun:test";
import { deriveServiceStatus } from "../../src/lib/service-status";

describe("deriveServiceStatus", () => {
  test("ready is connected, idle and green", () => {
    expect(deriveServiceStatus("ready")).toEqual({
      isActive: false,
      isConnected: true,
      isConnecting: false,
      dotClass: "bg-success",
    });
  });

  test("listening and processing are active and connected", () => {
    for (const status of ["listening", "processing"]) {
      const view = deriveServiceStatus(status);
      expect(view.isActive).toBe(true);
      expect(view.isConnected).toBe(true);
      expect(view.dotClass).toBe("bg-success");
    }
  });

  test("Connecting is amber and not connected", () => {
    const view = deriveServiceStatus("Connecting");
    expect(view.isConnecting).toBe(true);
    expect(view.isConnected).toBe(false);
    expect(view.dotClass).toBe("bg-warning");
  });

  test("error and unknown statuses are offline and red", () => {
    for (const status of ["error", "offline", ""]) {
      const view = deriveServiceStatus(status);
      expect(view.isActive).toBe(false);
      expect(view.isConnected).toBe(false);
      expect(view.isConnecting).toBe(false);
      expect(view.dotClass).toBe("bg-error");
    }
  });
});
