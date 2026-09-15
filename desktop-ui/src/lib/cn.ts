// SPDX-License-Identifier: GPL-3.0-or-later

export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}
