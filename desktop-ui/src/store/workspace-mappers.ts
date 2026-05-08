import type { DictionaryEntry } from '../features/dictionary/data/dictionary-data';
import {
  type AsrBackendValue,
  asrBackends,
  type DictationModeValue,
  deliveryMethods,
  labelFor,
} from '../features/settings/data/settings-data';
import type {
  BackendDictionaryEntry,
  BackendHistoryEntry,
  BackendWorkspaceSnapshot,
  SaveWorkspaceSettingsInput,
} from '../lib/backend/types';
import { DEFAULT_SETTINGS, type HistorySession, type WorkspaceSettings } from './workspace-types';

export function mapHistoryEntry(entry: BackendHistoryEntry): HistorySession | null {
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

export function mapDictionaryEntry(entry: BackendDictionaryEntry): DictionaryEntry {
  return entry;
}

export function mapWorkspaceSettings(snapshot: BackendWorkspaceSnapshot): WorkspaceSettings {
  return {
    activationMode: snapshot.settings.activationMode,
    asrBackend: snapshot.settings.asrBackendType,
    useGpu: snapshot.settings.useGpu,
    devicePolicy: snapshot.settings.devicePolicy,
    confidenceThreshold: formatThreshold(snapshot.settings.confidenceThreshold ?? 0.5),
    dictationMode: snapshot.settings.dictationMode,
    stylePreset: snapshot.settings.stylePreset,
    adaptiveVad: snapshot.settings.adaptiveVad,
    speechThreshold: formatThreshold(snapshot.settings.speechThreshold),
    speechThresholdEnd: formatThreshold(snapshot.settings.speechThresholdEnd ?? 0.3),
    hotkeyToggle: snapshot.settings.hotkeyToggle,
    hotkeyCancel: snapshot.settings.hotkeyCancel,
    pillVisibility: snapshot.settings.pillVisibility ?? 'Always',
    language: snapshot.settings.language ?? 'en',
    dictationLanguage: snapshot.settings.dictationLanguage ?? 'auto',
    micPrewarm: snapshot.settings.micPrewarm ?? DEFAULT_SETTINGS.micPrewarm,
    vadPcen: snapshot.settings.vadPcen ?? DEFAULT_SETTINGS.vadPcen,
    openaiAsrBaseUrl: snapshot.settings.openaiAsrBaseUrl ?? DEFAULT_SETTINGS.openaiAsrBaseUrl,
    openaiAsrApiKey: snapshot.settings.openaiAsrApiKeySet
      ? '••••••••'
      : DEFAULT_SETTINGS.openaiAsrApiKey,
    openaiAsrModel: snapshot.settings.openaiAsrModel ?? DEFAULT_SETTINGS.openaiAsrModel,
  };
}

export function summarizeAsrBackend(value: AsrBackendValue) {
  return labelFor(asrBackends, value).replace(' (recommended)', '').replace(' (fast)', '');
}

export function summarizeDictationMode(value: DictationModeValue) {
  return labelFor(deliveryMethods, value).replace(' into app', '');
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
    useGpu: settings.useGpu,
    devicePolicy: settings.devicePolicy,
    asrBackendType: settings.asrBackend,
    confidenceThreshold: Number(settings.confidenceThreshold),
    adaptiveVad: settings.adaptiveVad,
    speechThreshold: Number(settings.speechThreshold),
    speechThresholdEnd: Number(settings.speechThresholdEnd),
    stylePreset: settings.stylePreset,
    hotkeyToggle: settings.hotkeyToggle,
    hotkeyCancel: settings.hotkeyCancel,
    pillVisibility: settings.pillVisibility,
    language: settings.language,
    dictationLanguage: settings.dictationLanguage,
    micPrewarm: settings.micPrewarm,
    vadPcen: settings.vadPcen,
    openaiAsrBaseUrl: settings.openaiAsrBaseUrl || null,
    openaiAsrApiKey:
      settings.openaiAsrApiKey && !settings.openaiAsrApiKey.startsWith('•')
        ? settings.openaiAsrApiKey
        : null,
    openaiAsrModel: settings.openaiAsrModel || null,
  };
}
