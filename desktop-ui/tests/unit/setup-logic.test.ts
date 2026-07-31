import { describe, expect, test } from 'bun:test';
import {
  asrModelLocalName,
  buildInitialSelection,
  gpuAccelerationOffer,
  isSetupSatisfiable,
  type ModelStatus,
  relevantSetupModels,
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
    expect(asrModelLocalName('WhisperCpp', 'Small')).toBe('whisper-small-q5_1');
    expect(asrModelLocalName('WhisperCpp', 'Medium')).toBe('whisper-medium-q5_0');
    expect(asrModelLocalName('WhisperCpp', 'LargeV3Turbo')).toBe('whisper-large-v3-turbo-q8_0');
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
    // Usually a plugin installed this session that needs a restart.
    expect(gpuAccelerationOffer('failed', true)).toBe('after-download');
  });

  test('hides the card when no download could make the GPU work', () => {
    expect(gpuAccelerationOffer('unsupported', true)).toBeNull();
    expect(gpuAccelerationOffer('driver-missing', true)).toBeNull();
  });

  test('hides the card when the plugin archive is not published', () => {
    // Settings hides the download button in the same state, so pointing the
    // user there would be a dead end.
    expect(gpuAccelerationOffer('not-installed', false)).toBeNull();
    expect(gpuAccelerationOffer('failed', false)).toBeNull();
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
