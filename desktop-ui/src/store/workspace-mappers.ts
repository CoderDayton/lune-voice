// SPDX-License-Identifier: GPL-3.0-or-later

import type { I18n } from '@lingui/core';
import type { DictionaryEntry } from '../features/dictionary/data/dictionary-data';
import {
  type AsrBackendValue,
  asrBackendShortLabels,
  type DictationModeValue,
  dictationModeShortLabels,
} from '../features/settings/data/settings-data';
import { LIMITS } from '../lib/backend/generated/limits';
import type {
  DictionaryItemDto,
  HistoryItemDto,
  SaveWorkspaceSettingsInput,
  WorkspaceSnapshot,
} from '../lib/backend/types';
import { DEFAULT_SETTINGS, type HistorySession, type WorkspaceSettings } from './workspace-types';

export function mapHistoryEntry(entry: HistoryItemDto): HistorySession | null {
  if (!entry.transcript) return null;

  const startedAt = new Date(entry.startedAt);
  const endedAt = entry.endedAt ? new Date(entry.endedAt) : null;
  const seconds = endedAt
    ? Math.max(1, Math.round((endedAt.getTime() - startedAt.getTime()) / 1000))
    : 0;

  return {
    id: entry.id,
    startedAt: entry.startedAt,
    endedAt: entry.endedAt,
    transcript: entry.transcript,
    backend: entry.backend,
    style: entry.style,
    status: entry.status,
    words: countWords(entry.transcript),
    seconds,
  };
}

export function mapDictionaryEntry(entry: DictionaryItemDto): DictionaryEntry {
  return entry;
}

export function mapWorkspaceSettings(snapshot: WorkspaceSnapshot): WorkspaceSettings {
  return {
    activationMode: snapshot.settings.activationMode,
    asrBackend: snapshot.settings.asrBackendType,
    whisperModel: snapshot.settings.whisperModel,
    parakeetModel: snapshot.settings.parakeetModel ?? DEFAULT_SETTINGS.parakeetModel,
    useGpu: snapshot.settings.useGpu,
    devicePolicy: snapshot.settings.devicePolicy,
    confidenceThreshold: formatThreshold(
      snapshot.settings.confidenceThreshold ?? Number(DEFAULT_SETTINGS.confidenceThreshold),
    ),
    dictationMode: snapshot.settings.dictationMode,
    copyToClipboard: snapshot.settings.copyToClipboard ?? DEFAULT_SETTINGS.copyToClipboard,
    soundCuesEnabled: snapshot.settings.soundCuesEnabled ?? DEFAULT_SETTINGS.soundCuesEnabled,
    soundCueVolume: snapshot.settings.soundCueVolume ?? DEFAULT_SETTINGS.soundCueVolume,
    stylePreset: snapshot.settings.stylePreset,
    hotkeyToggle: snapshot.settings.hotkeyToggle,
    hotkeyCancel: snapshot.settings.hotkeyCancel,
    pillVisibility: snapshot.settings.pillVisibility ?? 'Always',
    pillAnchor: snapshot.settings.pillAnchor ?? DEFAULT_SETTINGS.pillAnchor,
    pillLocked: snapshot.settings.pillLocked ?? DEFAULT_SETTINGS.pillLocked,
    pillEdgeMargin: snapshot.settings.pillEdgeMargin ?? DEFAULT_SETTINGS.pillEdgeMargin,
    pillMonitor: snapshot.settings.pillMonitor ?? DEFAULT_SETTINGS.pillMonitor,
    pillAllWorkspaces: snapshot.settings.pillAllWorkspaces ?? DEFAULT_SETTINGS.pillAllWorkspaces,
    pillSize: snapshot.settings.pillSize ?? DEFAULT_SETTINGS.pillSize,
    pillOrientation: snapshot.settings.pillOrientation ?? DEFAULT_SETTINGS.pillOrientation,
    pillCorner: snapshot.settings.pillCorner ?? DEFAULT_SETTINGS.pillCorner,
    language: snapshot.settings.language ?? 'en',
    dictationLanguage: snapshot.settings.dictationLanguage ?? 'auto',
    modelIdleUnloadMinutes:
      snapshot.settings.modelIdleUnloadMinutes ?? DEFAULT_SETTINGS.modelIdleUnloadMinutes,
    inputDevice: snapshot.settings.inputDevice ?? DEFAULT_SETTINGS.inputDevice,
    agcMode: snapshot.settings.agcMode ?? DEFAULT_SETTINGS.agcMode,
    startMinimized: snapshot.settings.startMinimized ?? DEFAULT_SETTINGS.startMinimized,
    checkUpdatesOnLaunch:
      snapshot.settings.checkUpdatesOnLaunch ?? DEFAULT_SETTINGS.checkUpdatesOnLaunch,
    brandCasing: snapshot.settings.brandCasing ?? DEFAULT_SETTINGS.brandCasing,
    // No `??` fallback: `null` is the "Don't save history" choice, not a
    // missing value. Coercing it to the default would show the retention
    // select as "30 days" and silently re-enable history on the next save.
    sessionHistoryDays: snapshot.settings.sessionHistoryDays,
    terminalMode: snapshot.settings.terminalMode ?? DEFAULT_SETTINGS.terminalMode,
    privacyMode: snapshot.settings.privacyMode ?? DEFAULT_SETTINGS.privacyMode,
    llmEnabled: snapshot.settings.llmEnabled ?? DEFAULT_SETTINGS.llmEnabled,
    openaiAsrBaseUrl: snapshot.settings.openaiAsrBaseUrl ?? DEFAULT_SETTINGS.openaiAsrBaseUrl,
    openaiAsrApiKey: snapshot.settings.openaiAsrApiKeySet
      ? '••••••••'
      : DEFAULT_SETTINGS.openaiAsrApiKey,
    openaiAsrModel: snapshot.settings.openaiAsrModel ?? DEFAULT_SETTINGS.openaiAsrModel,
    // OpenAI-compatible is the only styling engine now — the provider fields
    // below persist regardless of `llmEnabled`. The key is masked when saved.
    llmBackend: 'OpenAiCompatible',
    openaiLlmBaseUrl: snapshot.settings.openaiLlmBaseUrl ?? DEFAULT_SETTINGS.openaiLlmBaseUrl,
    openaiLlmApiKey: snapshot.settings.openaiLlmApiKeySet
      ? '••••••••'
      : DEFAULT_SETTINGS.openaiLlmApiKey,
    openaiLlmModel: snapshot.settings.openaiLlmModel ?? DEFAULT_SETTINGS.openaiLlmModel,
  };
}

export function summarizeAsrBackend(i18n: I18n, value: AsrBackendValue) {
  const label = asrBackendShortLabels[value];
  return label ? i18n._(label) : value;
}

export function summarizeDictationMode(i18n: I18n, value: DictationModeValue) {
  const label = dictationModeShortLabels[value];
  return label ? i18n._(label) : value;
}

function countWords(text: string) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function formatThreshold(value: number) {
  return value.toFixed(2);
}

export function settingsToSaveInput(settings: WorkspaceSettings): SaveWorkspaceSettingsInput {
  return {
    activationMode: settings.activationMode,
    dictationMode: settings.dictationMode,
    copyToClipboard: settings.copyToClipboard,
    soundCuesEnabled: settings.soundCuesEnabled,
    soundCueVolume: settings.soundCueVolume,
    useGpu: settings.useGpu,
    devicePolicy: settings.devicePolicy,
    asrBackendType: settings.asrBackend,
    whisperModel: settings.whisperModel,
    parakeetModel: settings.parakeetModel,
    confidenceThreshold: Number(settings.confidenceThreshold),
    stylePreset: settings.stylePreset,
    hotkeyToggle: settings.hotkeyToggle,
    hotkeyCancel: settings.hotkeyCancel,
    pillVisibility: settings.pillVisibility,
    pillAnchor: settings.pillAnchor,
    pillLocked: settings.pillLocked,
    pillEdgeMargin: Math.max(
      0,
      Math.min(LIMITS.pillEdgeMarginMax, Math.round(settings.pillEdgeMargin) || 0),
    ),
    pillMonitor: settings.pillMonitor,
    pillAllWorkspaces: settings.pillAllWorkspaces,
    pillSize: settings.pillSize,
    pillOrientation: settings.pillOrientation,
    pillCorner: settings.pillCorner,
    language: settings.language,
    dictationLanguage: settings.dictationLanguage,
    modelIdleUnloadMinutes: settings.modelIdleUnloadMinutes,
    inputDevice: settings.inputDevice,
    agcMode: settings.agcMode,
    startMinimized: settings.startMinimized,
    checkUpdatesOnLaunch: settings.checkUpdatesOnLaunch,
    brandCasing: settings.brandCasing,
    sessionHistoryDays: settings.sessionHistoryDays,
    terminalMode: settings.terminalMode,
    privacyMode: settings.privacyMode,
    llmEnabled: settings.llmEnabled,
    openaiAsrBaseUrl: settings.openaiAsrBaseUrl || null,
    openaiAsrApiKey:
      settings.openaiAsrApiKey && !settings.openaiAsrApiKey.startsWith('•')
        ? settings.openaiAsrApiKey
        : null,
    openaiAsrModel: settings.openaiAsrModel || null,
    llmBackendType: settings.llmBackend,
    openaiLlmBaseUrl: settings.openaiLlmBaseUrl || null,
    openaiLlmApiKey:
      settings.openaiLlmApiKey && !settings.openaiLlmApiKey.startsWith('•')
        ? settings.openaiLlmApiKey
        : null,
    openaiLlmModel: settings.openaiLlmModel || null,
  };
}

// Display label for a `HistorySession.backend` string. Shared so the history
// table and the session detail modal cannot drift apart on what the same
// session's engine is called.
export function formatBackend(backend: string): string {
  if (backend.includes('WhisperCpp')) return 'Whisper';
  // "API", not "Cloud" — the endpoint may have been a server on this machine.
  if (backend.includes('OpenAi')) return 'API';
  return backend;
}
