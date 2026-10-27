// SPDX-License-Identifier: GPL-3.0-or-later

import type { MessageDescriptor } from '@lingui/core';
import { msg } from '@lingui/core/macro';
import {
  AudioLines,
  HardDriveDownload,
  type LucideIcon,
  Mic,
  Palette,
  Pill,
  Send,
  SlidersHorizontal,
  Wand2,
} from 'lucide-react';
import type { SettingsSectionId } from './sections';

export interface SettingsRailItem {
  id: SettingsSectionId;
  label: MessageDescriptor;
  icon: LucideIcon;
}

export interface SettingsRailGroup {
  name: MessageDescriptor;
  items: SettingsRailItem[];
}

/** The settings rail: every section, grouped by what it is for. */
export const settingsRailGroups: SettingsRailGroup[] = [
  {
    name: msg`App`,
    items: [
      { id: 'general', label: msg`General`, icon: SlidersHorizontal },
      { id: 'appearance', label: msg`Appearance`, icon: Palette },
      { id: 'pill', label: msg`Pill`, icon: Pill },
    ],
  },
  {
    name: msg`Speech`,
    items: [
      { id: 'dictation', label: msg`Dictation`, icon: Mic },
      { id: 'transcription', label: msg`Transcription`, icon: AudioLines },
      { id: 'models', label: msg`Models`, icon: HardDriveDownload },
    ],
  },
  {
    name: msg`Text`,
    items: [
      { id: 'styling', label: msg`Styling`, icon: Wand2 },
      { id: 'output', label: msg`Output`, icon: Send },
    ],
  },
];
