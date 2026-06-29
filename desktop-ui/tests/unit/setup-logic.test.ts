import { describe, expect, test } from 'bun:test';
import {
  asrModelLocalName,
  buildInitialSelection,
  isSetupSatisfiable,
  type ModelStatus,
} from '../../src/features/welcome/setup-logic';

const model = (localName: string, opts: Partial<ModelStatus> = {}): ModelStatus => ({
  name: localName,
  repoId: 'repo',
  localName,
  ready: false,
  description: '',
  required: false,
  ...opts,
});

describe('asrModelLocalName', () => {
  test('maps each Whisper size to its model directory', () => {
    expect(asrModelLocalName('WhisperCpp', 'Small')).toBe('whisper-small-q8_0');
    expect(asrModelLocalName('WhisperCpp', 'Medium')).toBe('whisper-medium-q8_0');
    expect(asrModelLocalName('WhisperCpp', 'LargeV3Turbo')).toBe('whisper-large-v3-turbo-q8_0');
  });
});

describe('buildInitialSelection', () => {
  test('always includes the chosen engine model', () => {
    const selected = buildInitialSelection('whisper-medium-q8_0', []);
    expect(selected.has('whisper-medium-q8_0')).toBe(true);
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
