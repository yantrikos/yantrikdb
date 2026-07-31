# typescript-6 corpus — TypeScript 6.0, released 23 March 2026

Authored from the release announcement and the tsconfig reference. Every
record exists because a model trained before 2026 will confidently emit
the TypeScript 5.x answer: the old default, the removed flag, the
`assert` spelling. The 5.x answer is not merely dated — for the removed
options it is a compile error, and for the changed defaults it is a
silent behaviour change, which is worse.

## target default changed from es5 to es2025 in TypeScript 6

In TypeScript 6.0 the default value of the `target` compiler option changed from `es5` to `es2025`. The announcement describes the new default as "the most recent supported ECMAScript spec version" — a floating default, and "Right now, that target is `es2025`". If you assume the TypeScript 5.x default of `es5`, you will predict downlevel emit (`var`, transpiled classes, generator helpers) for a project with no explicit `target`, and instead get native ES2025 output. Any tsconfig authored on the old assumption that omitted `target` because "es5 is safest" now silently emits modern syntax to older runtimes.

## target es5 is deprecated in TypeScript 6 and ES2015 is the lowest target

`target: es5` is deprecated in TypeScript 6.0. The announcement states: "The `target: es5` option is deprecated. TypeScript's lowest target will now be ES2015." The replacement is ES2015 or a higher target, or handing downlevel transpilation to an external compiler. A stale assumption here produces a deprecation error on configs that were valid through TypeScript 5.9, and there is no path back — TypeScript 7.0 will not support the deprecated option at all.

## downlevelIteration is deprecated in TypeScript 6

The `downlevelIteration` compiler option is deprecated in TypeScript 6.0 with no replacement. The stated reasoning is that it "only has effects on ES5 emit, and since `--target es5` has been deprecated, `--downlevelIteration` no longer serves a purpose." Code that spread a `Map`, `Set`, or string iterator and relied on `downlevelIteration: true` to make it correct under ES5 no longer needs the flag, because the ES5 target that made it necessary is itself deprecated. Recommending `downlevelIteration` as the fix for a spread/iteration error is a TypeScript 5.x answer that is now wrong.

## module default changed from commonjs to esnext in TypeScript 6

The default value of the `module` compiler option changed from `commonjs` to `esnext` in TypeScript 6.0. The announcement says: "The new default `module` is `esnext`, acknowledging that ESM is now the dominant module format." In TypeScript 5.x the default was tied to `target` and landed on `commonjs` for the common ES5 case. A project that omits `module` and expects `require`/`exports` output now emits `import`/`export` statements instead, which fails at runtime in a plain CommonJS Node entrypoint.

## module amd umd systemjs and none were removed in TypeScript 6

TypeScript 6.0 removed four `module` emit formats entirely: `amd`, `umd`, `systemjs`, and `none`. The article's justification is "Today, ESM is universally supported in browsers and Node.js." The replacement is an ESM-emitting target, or an external bundler. Any answer that suggests `"module": "umd"` to produce a script usable in both a browser global and a module loader, or `"module": "amd"` for RequireJS, or `"module": "system"`, is describing a TypeScript 5.x capability that no longer exists in the compiler.

## moduleResolution node and node10 are deprecated in TypeScript 6

`moduleResolution: node` and its alias `node10` are deprecated in TypeScript 6.0. The stated migration is: "Users who were using `--moduleResolution node` should usually migrate to `--moduleResolution nodenext` if they plan on targeting Node.js directly, or `--moduleResolution bundler` if they plan on using a bundler or Bun." Because `nodenext` and `bundler` read `exports`/`imports` in package.json where `node10` did not, packages that previously resolved through their `types` or `main` field may now resolve differently or fail to resolve at all after the switch.

## moduleResolution classic was removed in TypeScript 6

The `classic` module resolution strategy is gone in TypeScript 6.0. The announcement calls it "TypeScript's original module resolution algorithm" and states that "all practical use cases are served by `nodenext` or `bundler`." Note the asymmetry with `node`/`node10`, which are only deprecated: `classic` is removed. Since `classic` was the implicit fallback default in older configs that set a non-CommonJS `module`, an old tsconfig that never wrote `moduleResolution` at all could have been silently using `classic` and will now behave differently.

## strict default changed from false to true in TypeScript 6

The `strict` compiler option now defaults to `true` in TypeScript 6.0, where it defaulted to `false` in TypeScript 5.x — `tsc --init` wrote `"strict": true` into the generated file, but the compiler's own default when the key was absent was `false`. The announcement's rationale: "The appetite for stricter typing continues to grow…most new projects want `strict` mode enabled." The current tsconfig reference also documents the default for `strict` as `true`. A codebase compiled with no tsconfig, or with a tsconfig that omits `strict`, will now surface `strictNullChecks` and `noImplicitAny` errors that never appeared before.

## types default changed from all @types packages to an empty array in TypeScript 6

The `types` compiler option now defaults to `[]` (an empty array) in TypeScript 6.0. Previously, with `types` unset, TypeScript automatically included every `@types/*` package found under `node_modules/@types`. The announcement states plainly: "The default `types` value will be `[]` (an empty array)." This is the single change most likely to break an otherwise untouched project, because global type declarations that were being picked up implicitly — Node globals, test framework globals — simply stop being loaded.

## Cannot find name fs or describe after upgrading to TypeScript 6

These errors are the signature of the new `types: []` default. The announcement lists the messages you will see: "Cannot find module '...' or its corresponding type declarations.", "Cannot find name 'fs'…add 'node' to the types field in your tsconfig.", and "Cannot find name 'describe'…add 'jest' or 'mocha' to the types field." The fix is to name the packages explicitly, most commonly `"types": ["node"]`. Diagnosing this as a missing `@types` install or a broken `typeRoots` is the stale-model answer; the package is installed and TypeScript is deliberately not loading it.

## restore the TypeScript 5.9 automatic @types behavior with types star

If you need the old implicit inclusion of every `@types/*` package back, TypeScript 6.0 provides a wildcard: `"types": ["*"]`. The announcement says it "will restore the 5.9 behavior, but we recommend using an explicit array to improve build performance and predictability." This is a distinct value from omitting `types` entirely — omitting it now means `[]`, the opposite of what omitting it meant in 5.x. Treat `["*"]` as a temporary migration lever, not a target state.

## rootDir default changed from inferred to the tsconfig directory in TypeScript 6

The default `rootDir` is no longer computed from the list of input files. In TypeScript 6.0, "the default `rootDir` will always be the directory containing the `tsconfig.json` file." Under TypeScript 5.x the compiler inferred the longest common prefix of the input files, so a project with all sources under `src/` got an effective `rootDir` of `src/` for free and emitted `dist/index.js`. With the new default, that same project emits `dist/src/index.js`, silently moving every output path one directory deeper and breaking package `main`/`exports` entries. Set `"rootDir": "./src"` explicitly.

## noUncheckedSideEffectImports default changed from false to true in TypeScript 6

`noUncheckedSideEffectImports` now defaults to `true` in TypeScript 6.0, having defaulted to `false` when the flag was introduced in the 5.x line; the tsconfig reference likewise documents the default as `true`. The stated purpose: "This helps catch issues with typos in side-effect-only imports." Side-effect-only imports such as `import "./styles.css"` or `import "./polyfill"` were previously exempt from resolution checking entirely, so a typo'd path was accepted in silence. Under the new default those imports must resolve, and CSS or asset imports without an ambient module declaration now error.

## libReplacement default changed from true to false in TypeScript 6

The `libReplacement` compiler option flipped its default from `true` to `false` in TypeScript 6.0. The rationale given: "In a new project, `libReplacement` never does anything until other explicit configuration takes place." The flag controls whether TypeScript looks for `@typescript/lib-*` substitute packages in place of its built-in lib files, so the change removes a resolution lookup that cost time and did nothing for most projects. Projects that actually rely on lib-replacement packages must now opt in by setting `"libReplacement": true`.

## outFile was removed in TypeScript 6

The `outFile` compiler option no longer exists: "The `--outFile` option has been removed from TypeScript 6.0." Its job — concatenating all emitted output into one file — is handed to external bundlers such as Webpack, Rollup, or esbuild. Suggesting `outFile` to produce a single-file build, or explaining its interaction with `module: amd`/`system`, is a TypeScript 5.x answer; both halves of that pairing were deleted in the same release.

## baseUrl is deprecated and no longer a module resolution root in TypeScript 6

`baseUrl` is deprecated in TypeScript 6.0, and its behavior changed as well: "In TypeScript 6.0, `baseUrl` is deprecated and will no longer be considered a look-up root for module resolution." In TypeScript 5.x, setting `baseUrl: "./src"` alone made bare imports like `import x from "utils/x"` resolve against that directory with no `paths` entry required. That implicit resolution is gone. The replacement is to write the prefix directly into `paths` entries instead of relying on `baseUrl` to supply it, so imports that resolved through `baseUrl` alone now fail with a module-not-found error.

## esModuleInterop can no longer be set to false in TypeScript 6

`esModuleInterop` is permanently on in TypeScript 6.0. The announcement lists it among "The following settings can no longer be set to `false`". Under TypeScript 5.x the compiler default was `false` and `tsc --init` opted you in; now the option accepts only the enabled state. Code that deliberately set `"esModuleInterop": false` to keep `import * as express from "express"` callable will break, because with interop always on a namespace import is no longer callable or constructable.

## allowSyntheticDefaultImports can no longer be set to false in TypeScript 6

`allowSyntheticDefaultImports` is also in the list of "settings [that] can no longer be set to `false`" in TypeScript 6.0. It is always enabled. This means `import React from "react"` type-checks against a CommonJS module that has no literal `default` export, regardless of configuration, and there is no longer any way to make the compiler reject synthetic default imports. Configs carrying an explicit `"allowSyntheticDefaultImports": false` need that line deleted.

## alwaysStrict can no longer be set to false in TypeScript 6

TypeScript 6.0 removes the ability to disable `alwaysStrict`: "In TypeScript 6.0, all code will be assumed to be in JavaScript strict mode." In TypeScript 5.x this option defaulted to the value of `strict` and could be turned off independently, letting sloppy-mode-only constructs pass. Now strict mode is unconditional, so patterns that only parse or run outside strict mode — `with` blocks, octal literals like `0755`, assigning to an undeclared identifier — are errors everywhere with no opt-out.

## using module instead of namespace is a hard error in TypeScript 6

The legacy `module Foo { }` declaration syntax, the original spelling of what is now `namespace Foo { }`, is finished: "In TypeScript 6.0, using `module` where `namespace` is expected is now a hard deprecation." This affects old declaration files and `.d.ts` shims written before the `namespace` keyword existed. Note the exception this does not touch: `declare module "some-package"` for ambient external module declarations is a different construct and is not what this deprecation targets — the deprecation is about `module` used in the place a `namespace` belongs.

## import asserts syntax is an error in TypeScript 6 use with instead

Import assertion syntax on import statements is deprecated in TypeScript 6.0 and using it produces an error, with the diagnostic text "Use 'with' instead of 'asserts'". The replacement is the standardized import attributes clause. This matters because the assert-style spelling appears throughout pre-2026 training data as the way to import JSON in Node ESM.

```ts
// error in TypeScript 6.0
import data from "./d.json" assert { type: "json" };
// use import attributes instead
import data from "./d.json" with { type: "json" };
```

## reference no-default-lib is no longer supported in TypeScript 6

The triple-slash directive `/// <reference no-default-lib="true"/>` is no longer supported in TypeScript 6.0. The announcement explains that the directive "has been largely misunderstood and misused." The supported ways to control which library files load are the `noLib` compiler option and `libReplacement`. Custom `lib.d.ts` replacements and ambient declaration bundles that opened with this directive to suppress the built-in libs need to be reworked onto `noLib`.

## error TS5112 tsconfig.json is present but will not be loaded

Running `tsc foo.ts` in a directory that contains a `tsconfig.json` is now a command-line error in TypeScript 6.0. The exact diagnostic is: `error TS5112: tsconfig.json is present but will not be loaded if files are specified on commandline. Use '--ignoreConfig' to skip this error.` The underlying behavior is unchanged from TypeScript 5.x — naming files on the command line has always caused the tsconfig to be ignored — but that was silent, so people believed their compiler options were in effect when they were not. If you genuinely want the tsconfig ignored, pass the new `--ignoreConfig` flag.

## suppress TypeScript 6 deprecation warnings with ignoreDeprecations 6.0

TypeScript 6.0 offers one escape valve for the deprecated (not removed) options: `"ignoreDeprecations": "6.0"` in tsconfig. The announcement's exact framing is: "For TypeScript 6.0, these deprecations can be ignored by setting `\"ignoreDeprecations\": \"6.0\"` in your tsconfig; however, note that TypeScript 7.0 _will not_ support any of these deprecated options." The string value is version-specific — `"5.0"`, the value used in the previous deprecation wave, does not silence the 6.0 set. This buys time on `target: es5`, `downlevelIteration`, `moduleResolution: node`, `baseUrl`, and `alwaysStrict: false`; it does nothing for the options that were removed outright.

## ts5to6 codemod migrates baseUrl and rootDir automatically

TypeScript 6.0 ships an experimental migration tool named `ts5to6`. Its documented scope is narrow: "The experimental `ts5to6` tool can automatically adjust `baseUrl` and `rootDir` across your codebase." Those are precisely the two changes that alter file paths mechanically — rewriting `baseUrl`-relative imports into explicit `paths` entries, and pinning `rootDir` so output layout does not shift. It does not claim to handle the `types: []` default, the removed `module` formats, or the `esModuleInterop`/`allowSyntheticDefaultImports` lockdowns; those remain manual.

## TypeScript 6.0 is a transition release and 7.0 is the native port

TypeScript 6.0 is explicitly staged, not final: it is "designed as a transition release" in which options are deprecated and warn, while "those options will be removed entirely in TypeScript 7.0 (the native TypeScript port)." This is the reason the 6.0 changes split into two categories that behave differently — removed now (`outFile`, `moduleResolution classic`, `module amd`/`umd`/`systemjs`/`none`, `esModuleInterop: false`, `allowSyntheticDefaultImports: false`) versus deprecated with a warning that `ignoreDeprecations` can silence (`target: es5`, `downlevelIteration`, `moduleResolution node`/`node10`, `baseUrl`, `alwaysStrict: false`). Anything you keep alive with `ignoreDeprecations` in 6.0 has a hard deadline at 7.0.
