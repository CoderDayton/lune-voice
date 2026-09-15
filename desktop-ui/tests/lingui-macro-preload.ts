// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Compiles the Lingui macros away for `bun test`.
 *
 * Vite runs `@lingui/babel-plugin-lingui-macro` for the app build; bun test has
 * no such pass, and the macro's runtime throws on import. Without this, any
 * module reachable from a unit test could never use `msg` / `t` — which is
 * exactly where shared option labels live.
 *
 * Babel only runs the macro plugin here: types are left in place and the result
 * is handed back with a `ts`/`tsx` loader so bun strips them as usual.
 */
import { transformAsync } from '@babel/core';
import { plugin } from 'bun';

plugin({
  name: 'lingui-macro',
  setup(build) {
    build.onLoad({ filter: /\.tsx?$/ }, async (args) => {
      const source = await Bun.file(args.path).text();
      const isTsx = args.path.endsWith('.tsx');
      const loader = isTsx ? ('tsx' as const) : ('ts' as const);
      if (!source.includes('@lingui/core/macro') && !source.includes('@lingui/react/macro')) {
        return { contents: source, loader };
      }

      const result = await transformAsync(source, {
        filename: args.path,
        configFile: false,
        babelrc: false,
        plugins: ['@lingui/babel-plugin-lingui-macro'],
        parserOpts: { plugins: isTsx ? ['typescript', 'jsx'] : ['typescript'] },
      });

      return { contents: result?.code ?? source, loader };
    });
  },
});
