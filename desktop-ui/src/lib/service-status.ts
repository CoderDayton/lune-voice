// SPDX-License-Identifier: GPL-3.0-or-later

export interface ServiceStatusView {
  /** A dictation session is running (capturing or transcribing). */
  isActive: boolean;
  /** The runtime is reachable and can take a session. */
  isConnected: boolean;
  isConnecting: boolean;
  /** Dot colour: amber while connecting, green when reachable, red otherwise. */
  dotClass: 'bg-warning' | 'bg-success' | 'bg-error';
}

/**
 * The one reading of `runtime.serviceStatus` that every status indicator in
 * the shell shares, so the same state never shows two colours.
 */
export function deriveServiceStatus(status: string): ServiceStatusView {
  const isActive = status === 'listening' || status === 'processing';
  const isConnected = status === 'ready' || isActive;
  const isConnecting = status === 'Connecting';
  const dotClass = isConnecting ? 'bg-warning' : isConnected ? 'bg-success' : 'bg-error';
  return { isActive, isConnected, isConnecting, dotClass };
}
