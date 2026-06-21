# Vendored Box Pleating Studio (subset)

This is a **partial vendored copy** of [Box Pleating Studio][bps] by Mu-Tsun
Tsai, used headlessly to run its packing optimizer and layout core when
generating box-pleated synthetic data. It is checked in so the generator does
not depend on an out-of-tree clone (previously `/tmp/bp-studio-source`, which a
system temp-cleanup wiped).

- **Source:** https://github.com/MuTsunTsai/box-pleating-studio
- **Commit:** `d86f8051812458b2c7b1ed7fac49fe7dc1d4dad4` (tag `v0.7.14`)
- **License:** MIT — see `LICENSE.md` (preserved from upstream).

## What is included (and why)

Only the subset our headless scripts load at runtime:

- `lib/optimizer/debug/` — the optimizer `optimizer.js` + `optimizer.wasm`. The
  `debug` build is the one that runs under Node/Bun; the upstream `dist` /
  `dist_mp` builds are browser/worker-only (they reference `self.location`) and
  are intentionally **not** vendored.
- `lib/optimizer/types.d.ts`, `lib/optimizer/debug/optimizer.d.ts` — type
  declarations referenced by the optimizer module.
- `src/core/`, `src/shared/` — the design/layout core imported via
  `NODE_PATH=<root>/src` as `core/...` and `shared/...`. These are the only
  source roots reachable from our entry modules (`core/design/context/tree`,
  `core/design/tasks/height`, `core/service/{processor,state,updateResult}`).

The rest of the upstream repo (app/UI, build tooling, tests, assets) is not
needed and is omitted.

## How it is wired

`src/bp-studio-optimizer.ts` and `src/bp-studio-layout.ts` default
`BP_STUDIO_ROOT` to this directory. Override with the `BP_STUDIO_ROOT`
environment variable or `--bp-studio-root` to point at a full clone.

## Updating

Re-clone upstream at the desired commit and recopy the same subset:

```bash
git clone https://github.com/MuTsunTsai/box-pleating-studio /tmp/bps
V=tools/synthetic-generator/vendor/box-pleating-studio
cp /tmp/bps/lib/optimizer/debug/optimizer.{js,wasm,d.ts} "$V/lib/optimizer/debug/"
cp /tmp/bps/lib/optimizer/types.d.ts "$V/lib/optimizer/"
rm -rf "$V/src/core" "$V/src/shared"
cp -R /tmp/bps/src/core /tmp/bps/src/shared "$V/src/"
cp /tmp/bps/LICENSE.md "$V/LICENSE.md"
```

Then bump the commit/tag recorded above.

[bps]: https://github.com/MuTsunTsai/box-pleating-studio
