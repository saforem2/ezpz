# Hyperframes Composition Brief: ezpz

## Objective
Deadpan startup-pitch brag video for `ezpz` (v0.27.3, distributed PyTorch launcher). Fake serious Series A energy applied to real HPC boilerplate.

## Output
- Directory: `brag-output-20260918-121350/composition/`
- Rendered: `brag-output-20260918-121350/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20s (trimmed from 21.5s storyboard)

## Source Material
- Root: `/Users/sam/projects/saforem2/ezpz/`
- Files read: `README.md`, `pyproject.toml`, `src/ezpz/__about__.py` (v0.27.3), `SKILL.md`, docs/
- Product: `ezpz` (lemon emoji 🍋 branding)
- Copy that must appear verbatim (from README / source):
  - `import ezpz`
  - `setup_torch()` / `get_torch_device()` / `wrap_model()`
  - `ezpz launch python3 train.py`
  - `ezpz submit -N 2 -q debug -- python3 train.py`
  - `"WRITE ONCE."` / `"RUN ANYWHERE."` (headline claims — paraphrased from README tagline)
  - `ezpz.cool`

## Creative Direction
- Tone preset: `yc-parody`
- Direction: "fake serious Series A pitch for a distributed PyTorch launcher on supercomputers"
- Angle: Apply startup-launch hyperbole (massive claims, stat ladders, deadpan delivery) to genuinely unsexy but impressive HPC features. Humor from framing, not from altering product claims.
- Hook: `"WRITE ONCE."` + `import ezpz` typing — startup bravado meeting actual Python import.
- Outro: `"RUN ANYWHERE."` + 🍋 logo + `ezpz.cool` with a single dry bell payoff.
- Avoid: generic SaaS language (no "streamline workflow"); no abstract filler visuals; no chaotic flashing (restraint is the joke).

## Visual Identity
- Background: `#0b0f14` (near-black terminal/card)
- Text: `#f4f4f5` off-white (mono) / `#f0d83a` yellow accent (lemon)
- Accent bar: `#f0d83a` (lemon from 🍋)
- Display font: `Playfair Display` (big serif headlines — pitch energy)
- Body/code font: `JetBrains Mono` (real source snippet recreation)
- References: README lemon branding; CLI `ezpz launch` / `ezpz submit`; `setup_torch()` snippet; docs color scheme (dark + yellow)

## Storyboard (contract — durations sum to ~20s; trim Scene 4 to 3.5s)
1. HOOK "import ezpz" — 2.5s — monospace typing → hard cut to serif `"WRITE ONCE."`
2. THE CLAIM stat ladder — 5s — 3 sequential rows (`Laptop` → `1 GPU` → `1,000 nodes — same command`) + ghosted CLI line
3. THE FLOW code snippet — 6s — 4 sequential code lines (`setup_torch` → `get_torch_device` → `Linear` → `wrap_model`)
4. THE COMMAND launch/submit — 3.5s (trimmed) — 2 CLI lines sequential (`launch` → `submit`)
5. OUTRO logo + `"RUN ANYWHERE."` — 3.5s — headline hold → logo + `ezpz.cool` + single bell

## Audio
- Music assets: **MISSING** in this environment (`~/.claude/skills/brag/assets/music/` and `skills/brag/assets/music/` both absent). Per pitfall: skip the audio bed cleanly — no fabricated track, no broken `<audio src>`. Documented below; render proceeds without bed.
- Music cue guidance: `unavailable; continue without beat/cue sync.` No preset, no `analyze_music_cues.py`, no `npx hyperframes beats`. No beat-locked reveals required.
- Audio-reactive treatment: none (no music = nothing to react to). If audio were present: subtle glow on lemon logo only.
- Audio-coupled moments:
  - Scene 1 — typing animation + keyboard ticks (`keypress-*` randomized)
  - Scene 2 — 3 stat rows reveal; accent first + final only (restraint)
  - Scene 3 — 4 code lines reveal sequentially; keyboard ticks
  - Scene 5 — logo appearance locks to single `impactBell_heavy_000` (the only big sound; payoff only)
- SFX selection (reference only — Hyperframes selects exact files):
  - `interface/drop_001.ogg` or `impactSoft_medium_001.ogg` (stat reveal, dry)
  - `keyboard/keypress-002/008/015.wav` (typing; randomized)
  - `impactBell_heavy_000.ogg` (logo payoff; use once)
- SFX analysis: `skills/brag/assets/sfx/sfx-analysis.md` unavailable locally; rely on audio.md guidance above (low/mid HF risk preferred for restrained tone).
- Music file copied: **NONE** (assets missing — clean skip, no broken src).
- Restraint rule: no dense rhythm, no voiceover (`--voice` not enabled), no waveform visuals. The parody is delivered by deadpan visuals + real product copy, not by sound energy.

## Hyperframes Instructions
- Load `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes`, `hyperframes-cli`. /brag is its own workflow — do not enter the generic hyperframes entry-point intent interview or promo/video workflow.
- Build standalone composition (no `<template>` wrapper on root); root has `data-composition-id="ezpz-brag"`, `data-width="1920"`, `data-height="1080"`.
- Define `window.__timelines["ezpz-brag"]` with a `gsap.timeline({ paused: true })`.
- CSS keyframe timeline must control duration (not just `data-duration`). Define sequential `@keyframes` windows for 5 scenes over 20s (see `hyperframes-workflow.md` snippet). Each scene has `.clip` class.
- No music bed present → no `<audio>` element needed; but if any SFX are added, each must have a unique `id` and proper `data-start`/`data-duration`/`data-track-index`/`data-volume`.
- At least one scene must recreate real source material: Scene 3 (the `setup_torch()` snippet) is the mandatory real-product scene per Q4.
- Typography: use actual monospace for code lines (`font-family: 'JetBrains Mono', monospace`) and serif for headlines (`Playfair Display`, with `@font-face` to a local file or a web-safe fallback). Per lint: a named font-family requires an in-file `@font-face` or `lint` fires `font_family_without_font_face` — either embed the font or declare a web-safe fallback.
- `data-hidden` not needed; hidden scenes managed by `.clip` opacity windows.
- Before render: run `npx hyperframes check`. 0 lint errors required. A runtime duration-probe error (`check_runtime_failure`) is non-blocking per pitfall.
- Render: `npx hyperframes render . -o ../brag.mp4 -f 30`
