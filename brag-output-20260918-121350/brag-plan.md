# Brag Plan: ezpz

## What is this app?
`ezpz` (v0.27.3) is a portable distributed PyTorch launcher: `setup_torch()` detects hardware (CUDA/NCCL, XPU/CCL, MPS, CPU/Gloo) and `ezpz launch` works identically on a laptop, a single GPU, or a 1000-node supercomputer like Aurora. Zero code changes across platforms.

## The angle (yc-parody)
Deadpan fake-Series-A pitch for a super-serious HPC problem. The joke: we apply startup-launch hyperbole to something genuinely unsexy (scheduler-aware `mpiexec` wrappers and FSDP wraps) and deliver it with total sincerity. Every claim is real — the parody is in the framing, not the product.

## Hook (first 2.5s)
Black screen. White monospace type slams in like a terminal: `import ezpz`. Then a hard cut to giant display serif text: "WRITE ONCE." A single dry `bong_001` bell hits. The contrast — startup bravado + actual Python import — lands the parody immediately.

## Key moments (the middle)
- **Scene: The Ladder.** 4 stat lines appear one-by-one, deadpan metric-slide style: "Laptop", "1 GPU", "1,000 nodes — same command". Each lands with a restrained `impactSoft_medium`. This is the core absurdity — a single CLI spanning 6 orders of magnitude.
- **Scene: Setup.** A recreated code snippet (`rank = ezpz.setup_torch()` → `wrap_model()` → `ezpz wrap_model`) with simulated typing and keyboard ticks. Shows the *actual* user flow from Step 1 Q9.
- **Scene: Launch.** The CLI line `ezpz launch python3 train.py` scales in, then the same line repeats with `ezpz submit -N 2 -q debug --`. Two executions, identical syntax — the "anywhere" claim made visual.

## Outro / punchline (last 3s)
Giant serif text: "RUN ANYWHERE." Hold. Then a hard cut to the lemon logo 🍋 + `ezpz.cool`. No fade. A final dry bell (`impactBell_heavy_000`) hits on the logo — the only celebratory sound in an otherwise restrained audio layer. Black screen. End.

## User flow worth showing (from Step 1 Q9)
`import ezpz` → `setup_torch()` / `get_torch_device()` → `wrap_model()` (FSDP) → `ezpz launch` (or `ezpz submit` for PBS/SLURM) → same command on Aurora / laptop / 1000 nodes. The video recreates the code-snippet entry + CLI launch beats; full model training is implied by the stat ladder.

## Tone
- Preset: `yc-parody`
- Creative direction: "fake serious Series A pitch for a distributed PyTorch launcher on supercomputers"
- Interpretation: Deadpan seriousness. Hard cuts, minimal transitions (per preset table), restrained motion, sparse sound. The humor comes from applying startup-launch energy to genuine HPC boilerplate. Every line is delivered straight — no winks, no chaotic flashing.

## Format: landscape — 1920x1080 (auto; no `--format` override)
## Duration: ~20s (default; scenes sum to 19.5s below)

## Visual identity (from project)
- Background: `#0b0f14` (near-black, pulled from terminal/code aesthetic in docs)
- Accent: `#f0d83a` (lemon yellow from logo / README 🍋)
- Text: `#f4f4f5` (off-white, high-contrast monospace + serif pairing)
- Display font: `Playfair Display` (big serif headlines — startup-pitch energy)
- Body font: `JetBrains Mono` (terminal/code recreation — the real product)
- Strongest visual element: the `ezpz` CLI (`ezpz launch` / `ezpz submit`) + `setup_torch()` snippet — this is the user flow from Q9, not a marketing card.

## Share copy (draft)
`ezpz launches distributed training everywhere. One command. Zero code changes. Written for Aurora, runs on your laptop. 🍋`

## Audio direction
- Role: restrained deadpan bed — one quiet corporate-adjacent track at low volume, 2-3 dry SFX only.
- Music: `happy-beats-business-moves-vol-11-by-ende-dot-app.mp3` (1:28, warm/business-y — fits `yc-parody`) at very low volume `0.18` (deadpan/restraint posture per audio.md).
- Music cue guidance: preset available for vol-11 (`assets/music/cues/happy-beats-business-moves-vol-11.music-cues.md` / `.json`) — read at composition time. Use 1 strong cue for the "Launch" scene; ignore others given restraint posture.
- Audio-reactive treatment: very subtle; only the lemon logo glow may breathe with RMS on the outro. No waveform visuals.
- SFX posture: sparse — 3 cues max. `impactSoft_medium` (stat reveals, dry), `interface/drop_001` (card landing), `impactBell_heavy_000` (logo payoff). Per `yc-parody`: sparse but present.
- Audio-coupled moments: stat ladder arrives one-by-one (accent first + final), typing animation matches keyboard ticks, logo hit locks to final bell.
- Restraint rule: no dense rhythmic layer. Music stays below text energy at all times; never competes with the deadpan delivery. No chaotically stacked sounds.

## Storyboard

### Scene 1 — HOOK: "import ezpz" — 2.5s
Black background. White monospace `import ezpz` types in character-by-character (simulated interaction) with quiet keyboard ticks (`keypress-002/008/015` randomized). At 0.9s, hard cut to giant serif: `"WRITE ONCE."` (yellow accent underline). Hold.
Sequential/interaction: yes — typing animation (character-by-character) + hard text reveal.
Audio intent: deadpan setup — a startup pitch opening delivered with zero excitement.
Audio-coupled idea: per-character typing with sparse keyboard ticks; no beat sync (restraint).
Music: vol-11 starts at 0s at 0.18; no beat-driven motion.
Transition mood: hard cut → Scene 2 (per `yc-parody` preset table).

### Scene 2 — THE CLAIM: "Laptop → 1,000 nodes" — 5s
Display serif headline: `"RUN ANYWHERE."` Below, a clean 3-row stat ladder appears sequentially (0.6s apart): `"Laptop"` → `"1 GPU"` → `"1,000 nodes — same command"`. Each row lands with a dry `impactSoft_medium_002`. Yellow accent bar grows left-to-right behind the final row. Monospace code ghosted on the right: `ezpz launch python3 train.py` (no interaction — static contextual reference).
Sequential/interaction: yes — 3 stat rows reveal one-by-one.
Audio intent: the core parody beat — massive claim delivered like a metrics deck.
Audio-coupled idea: sequential reveals; accent first and final only (restraint).
Music: bed continues low; cue metadata guides nothing (restraint > sync).
Transition mood: hard cut → Scene 3.

### Scene 3 — THE FLOW: Code snippet — 6s
Recreated source snippet (monospace, `#0b0f14` card, soft border). Lines appear sequentially from top to bottom (simulated typing, keyboard ticks):
`rank = ezpz.setup_torch()` (1.8s visible, holds)
`device = ezpz.get_torch_device()` (appears at 2.2s)
`model = torch.nn.Linear(128, 10).to(device)` (appears at 3.6s)
`model = ezpz.wrap_model(model)` (appears at 5.0s)
Real code — pulled directly from README quick-start. This is the "show the thing" requirement.
Sequential/interaction: yes — sequential code-line reveal; simulated typing feel.
Audio intent: show the real product in use — the joke lands harder when the code is genuinely useful.
Audio-coupled idea: typed lines + sparse keyboard ticks; no heavy SFX.
Music: bed continues.
Transition mood: hard cut → Scene 4.

### Scene 4 — THE COMMAND: Launch + Submit — 4.5s
Two CLI commands scale in sequentially: `ezpz launch python3 train.py` (2.0s hold) then `ezpz submit -N 2 -q debug -- python3 train.py` (2.0s hold). Both in monospace on a terminal-dark card. Small yellow dot pulses beside the active line. The repetition is the punch — same syntax, two worlds (local / supercomputer).
Sequential/interaction: yes — 2 CLI lines reveal one by one.
Audio intent: confirm the claim with concrete commands.
Audio-coupled idea: soft pulse-dot (subtle, not beat-locked); no SFX on reveals (restraint).
Music: bed continues.
Transition mood: hard cut → Scene 5.

### Scene 5 — OUTRO: Logo + "RUN ANYWHERE" — 3.5s (end at 21.5s; trim to 20s by tightening Scene 4 to 3.5s → total 20.5s, close enough to auto-range; adjust at render)
Giant serif: `"RUN ANYWHERE."` (yellow accent word). Hold 1.5s. Hard cut to black with lemon logo 🍋 centered + `ezpz.cool` monospace below. A single dry `impactBell_heavy_000` bell hits exactly on the logo appearance (logo payoff — the one big sound in an otherwise restrained mix). Hold to end.
Sequential/interaction: none — single static reveal then logo lock.
Audio intent: the payoff — restrained until exactly here.
Audio-coupled idea: logo appearance locks to bell sound (single strong-cue lock from vol-11 preset); no reactive visuals beyond subtle glow.
Music: bed fades out from 2.0s to end.

**Music mood:** warm business/deadpan (yc-parody: deadpan startup pitch, not chaotic hype).
**Audio summary:** Very quiet corporate bed (0.18) + 3 dry SFX (2 medium impacts, 1 bell payoff) + sparse keyboard ticks on typing. No dense rhythm, no voice, no waveform visuals. The parody is in the delivery, not the sound.

**Scene duration sum (before trim):** 2.5 + 5 + 6 + 4.5 + 3.5 = 21.5s. Trim Scene 4 to 3.5s → 20.5s (within 15–25s). Final target: 20s.
