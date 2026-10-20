// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, test } from 'bun:test';
import {
  asrModelLocalName,
  buildInitialSelection,
  gpuAccelerationOffer,
  hasOptionalModels,
  initialSetupEngine,
  initialWhisperModel,
  isSetupSatisfiable,
  type ModelStatus,
  relevantSetupModels,
  shouldInstallGpuDuringSetup,
} from '../../src/features/welcome/setup-logic';

const model = (localName: string, opts: Partial<ModelStatus> = {}): ModelStatus => ({
  name: localName,
  repoId: 'repo',
  localName,
  ready: false,
  description: '',
  required: false,
  inUse: false,
  ...opts,
});

describe('asrModelLocalName', () => {
  test('maps each Whisper size to its model directory', () => {
    expect(asrModelLocalName('WhisperCpp', 'Small', 'TdtV3Q8')).toBe('whisper-small-q5_1');
    expect(asrModelLocalName('WhisperCpp', 'Medium', 'TdtV3Q4K')).toBe('whisper-medium-q5_0');
    expect(asrModelLocalName('WhisperCpp', 'LargeV3Turbo', 'TdtV3Q8')).toBe(
      'whisper-large-v3-turbo-q8_0',
    );
  });

  test('Parakeet installs its chosen size whatever Whisper size is saved', () => {
    expect(asrModelLocalName('ParakeetCpp', 'Small', 'TdtV3Q8')).toBe('parakeet-tdt-0.6b-v3-q8_0');
    expect(asrModelLocalName('ParakeetCpp', 'LargeV3Turbo', 'TdtV3Q4K')).toBe(
      'parakeet-tdt-0.6b-v3-q4_k',
    );
  });

  test('a remote engine never leaves setup without a model to install', () => {
    expect(asrModelLocalName('OpenAiCompatible', 'Medium', 'TdtV3Q8')).toBe('whisper-medium-q5_0');
  });
});

describe('initialSetupEngine', () => {
  test('a saved Parakeet choice survives re-running setup', () => {
    expect(initialSetupEngine('ParakeetCpp')).toBe('ParakeetCpp');
  });

  test('Whisper and remote engines start setup on Whisper', () => {
    expect(initialSetupEngine('WhisperCpp')).toBe('WhisperCpp');
    expect(initialSetupEngine('OpenAiCompatible')).toBe('WhisperCpp');
  });
});

describe('initialWhisperModel', () => {
  const noneInstalled = [model('whisper-medium-q5_0'), model('whisper-small-q5_1')];

  test('a machine with a GPU keeps the balanced default', () => {
    expect(initialWhisperModel('Medium', noneInstalled, { gpu: true, cpuThreads: 4 })).toBe(
      'Medium',
    );
  });

  test('a CPU-only machine with enough threads starts on Small', () => {
    expect(initialWhisperModel('Medium', noneInstalled, { gpu: false, cpuThreads: 8 })).toBe(
      'Small',
    );
  });

  test('a low-end CPU-only machine starts on Base', () => {
    expect(initialWhisperModel('Medium', noneInstalled, { gpu: false, cpuThreads: 4 })).toBe(
      'Base',
    );
  });

  test('an installed Whisper model means the saved choice stands', () => {
    // Past the first run the user has chosen; re-running setup must not
    // quietly switch them to a different model.
    const installed = [model('whisper-medium-q5_0', { ready: true }), model('whisper-small-q5_1')];
    expect(initialWhisperModel('Medium', installed, { gpu: false, cpuThreads: 2 })).toBe(
      'Medium',
    );
  });
});

describe('hasOptionalModels', () => {
  test('one required engine model is not a choice', () => {
    // The list the setup step actually gets today. Showing it under "choose the
    // models to install" costs a click and offers nothing back, so this is what
    // decides the step is skipped.
    const models = [model('whisper-medium-q5_0', { inUse: true })];

    expect(hasOptionalModels('whisper-medium-q5_0', models)).toBe(false);
  });

  test('a model that is neither the engine nor required is a choice', () => {
    const models = [model('whisper-medium-q5_0', { inUse: true }), model('extra')];

    expect(hasOptionalModels('whisper-medium-q5_0', models)).toBe(true);
  });

  test('a model flagged required cannot be declined, so it is not a choice', () => {
    const models = [model('whisper-medium-q5_0', { inUse: true }), model('vad', { required: true })];

    expect(hasOptionalModels('whisper-medium-q5_0', models)).toBe(false);
  });
});

describe('buildInitialSelection', () => {
  test('always includes the chosen engine model', () => {
    const selected = buildInitialSelection('whisper-medium-q5_0', []);
    expect(selected.has('whisper-medium-q5_0')).toBe(true);
  });

  test('includes installed and required models, skips the rest', () => {
    const models = [
      model('ready-one', { ready: true }),
      model('required-one', { required: true }),
      model('optional-one'),
    ];
    const selected = buildInitialSelection('engine', models);
    expect([...selected].sort()).toEqual(['engine', 'ready-one', 'required-one']);
  });
});

describe('relevantSetupModels', () => {
  test('keeps only the in-use models from the full library', () => {
    const models = [
      model('whisper-small-q5_1'),
      model('whisper-medium-q5_0', { inUse: true }),
      model('whisper-large-v3-turbo-q8_0'),
    ];
    const relevant = relevantSetupModels(models).map((m) => m.localName);
    expect(relevant).toEqual(['whisper-medium-q5_0']);
  });
});

describe('gpuAccelerationOffer', () => {
  test('claims acceleration is active only when a CUDA device is registered', () => {
    expect(gpuAccelerationOffer('ready', true)).toBe('active');
    // Already working, so it stays offered even where no download is available.
    expect(gpuAccelerationOffer('ready', false)).toBe('active');
  });

  test('promises setup rather than speed while the plugin is missing', () => {
    expect(gpuAccelerationOffer('not-installed', true)).toBe('after-download');
  });

  test('asks for a restart instead of a download once the plugin is on disk', () => {
    expect(gpuAccelerationOffer('needs-restart', true)).toBe('needs-restart');
    // Nothing left to fetch, so an unpublished archive changes nothing.
    expect(gpuAccelerationOffer('needs-restart', false)).toBe('needs-restart');
  });

  test('hides the card when no download could make the GPU work', () => {
    expect(gpuAccelerationOffer('unsupported', true)).toBeNull();
    expect(gpuAccelerationOffer('driver-missing', true)).toBeNull();
    expect(gpuAccelerationOffer('driver-outdated', true)).toBeNull();
    // `failed` is a driver too old for the runtime; re-downloading the plugin
    // does not fix it, and promising setup that cannot finish is worse than
    // saying nothing.
    expect(gpuAccelerationOffer('failed', true)).toBeNull();
  });

  test('hides the card when the plugin archive is not published', () => {
    // Settings hides the download button in the same state, so pointing the
    // user there would be a dead end.
    expect(gpuAccelerationOffer('not-installed', false)).toBeNull();
  });
});

describe('shouldInstallGpuDuringSetup', () => {
  test('downloads only when the plugin is genuinely absent', () => {
    expect(shouldInstallGpuDuringSetup('not-installed', true, true)).toBe(true);
  });

  test('never downloads over a declined GPU', () => {
    expect(shouldInstallGpuDuringSetup('not-installed', true, false)).toBe(false);
  });

  test('never downloads an archive this build has not published', () => {
    expect(shouldInstallGpuDuringSetup('not-installed', false, true)).toBe(false);
  });

  test('re-downloads nothing for a state a download cannot fix', () => {
    // The plugin is already on disk in both: waiting for a restart, or facing
    // a driver too old for it. Fetching it again helps with neither.
    expect(shouldInstallGpuDuringSetup('needs-restart', true, true)).toBe(false);
    expect(shouldInstallGpuDuringSetup('failed', true, true)).toBe(false);
    expect(shouldInstallGpuDuringSetup('ready', true, true)).toBe(false);
    expect(shouldInstallGpuDuringSetup('driver-missing', true, true)).toBe(false);
    expect(shouldInstallGpuDuringSetup('driver-outdated', true, true)).toBe(false);
    expect(shouldInstallGpuDuringSetup('unsupported', true, true)).toBe(false);
  });
});

describe('isSetupSatisfiable', () => {
  test('false until the engine model is selected', () => {
    expect(isSetupSatisfiable('engine', [], new Set())).toBe(false);
    expect(isSetupSatisfiable('engine', [], new Set(['engine']))).toBe(true);
  });

  test('requires every backend-required model to be selected', () => {
    const models = [model('engine', { required: true }), model('llm', { required: true })];
    expect(isSetupSatisfiable('engine', models, new Set(['engine']))).toBe(false);
    expect(isSetupSatisfiable('engine', models, new Set(['engine', 'llm']))).toBe(true);
  });

  test('ignores optional models', () => {
    const models = [model('engine'), model('optional')];
    expect(isSetupSatisfiable('engine', models, new Set(['engine']))).toBe(true);
  });
});
