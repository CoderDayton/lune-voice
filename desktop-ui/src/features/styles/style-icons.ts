import type { LucideIcon } from 'lucide-react';
import { Briefcase, Code2, Feather, MessageCircle, PenTool, Scroll, Sparkles } from 'lucide-react';
import type { StylePreset } from '../../store/workspace-types';

const STYLE_ICONS: Record<string, LucideIcon> = {
  minimal: Feather,
  casual: MessageCircle,
  professional: Briefcase,
  formal: Scroll,
  technical: Code2,
  writing: PenTool,
};

/** A representative icon for a style preset; custom styles fall back to Sparkles. */
export function styleIcon(preset: StylePreset): LucideIcon {
  return STYLE_ICONS[preset.id] ?? Sparkles;
}
