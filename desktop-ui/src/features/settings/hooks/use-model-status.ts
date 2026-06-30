import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useCallback, useEffect, useState } from 'react';
import { isTauriRuntime } from '../../../lib/backend/runtime';

export interface ModelStatus {
  name: string;
  repoId: string;
  localName: string;
  ready: boolean;
  description: string;
  required: boolean;
  /** Whether the active backend selection currently uses this model. */
  inUse: boolean;
  /** Pipeline role for library grouping: `'speech-to-text'` | `'styling'`. */
  role: string;
  /** Actual on-disk size in bytes when installed; `null` otherwise. */
  sizeBytes: number | null;
}

export interface DownloadProgress {
  modelName: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  done: boolean;
}

export interface ModelStatusApi {
  models: ModelStatus[];
  /** Lookup by on-disk directory (`local_name`). */
  byName: Map<string, ModelStatus>;
  loading: boolean;
  /** `local_name` currently downloading, or `null`. */
  downloading: string | null;
  progress: DownloadProgress | null;
  refresh: () => void;
  download: (localName: string) => void;
  remove: (localName: string) => Promise<void>;
}

/**
 * Shared model lifecycle: lists on-device model status, streams download
 * progress, and downloads/deletes by `local_name`. Used by the Models tab and
 * the Transcription engine cards so the two never drift on download wiring.
 *
 * `check_model_status` reports the full installable library (every Whisper size
 * plus the styling LLM), each flagged with `inUse` for the active config. Call
 * `refresh` when the backend or selected model changes to keep `inUse` current.
 */
export function useModelStatus(): ModelStatusApi {
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);

  const refresh = useCallback(() => {
    if (!isTauriRuntime()) {
      setLoading(false);
      return;
    }
    invoke<ModelStatus[]>('check_model_status')
      .then(setModels)
      .catch((e) => console.error('check_model_status:', e))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    const unlisten = listen<DownloadProgress>('lune://download-progress', (event) => {
      setProgress(event.payload);
      if (event.payload.done) {
        setDownloading(null);
        setProgress(null);
        refresh();
      }
    });
    return () => {
      void unlisten.then((f) => f());
    };
  }, [refresh]);

  const download = useCallback((localName: string) => {
    setDownloading(localName);
    setProgress(null);
    invoke('download_model_cmd', { localName }).catch((e) => {
      console.error('download_model_cmd:', e);
      setDownloading(null);
      setProgress(null);
    });
  }, []);

  const remove = useCallback(
    async (localName: string) => {
      try {
        await invoke('delete_model_cmd', { localName });
        refresh();
      } catch (e) {
        console.error('delete_model_cmd:', e);
      }
    },
    [refresh],
  );

  const byName = new Map(models.map((m) => [m.localName, m]));

  return { models, byName, loading, downloading, progress, refresh, download, remove };
}
