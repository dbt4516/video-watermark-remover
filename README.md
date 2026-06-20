---
name: video-watermark-removal
description: "Remove static watermarks/logos from video. Auto-detects watermark position, selects best method (bg-fill for solid backgrounds, ProPainter AI for complex textures). Pure Python, no vision model needed."
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [video, ffmpeg, watermark, computer-vision, media]
---

# Video Watermark Removal

Remove a static (non-moving) watermark or logo burned into a video file.

## Quick Start

```bash
python3 scripts/watermark_remover.py VIDEO.mp4
```

That's it. The script auto-detects the watermark, selects the best removal
method, processes the video, and outputs `<name>_nowm.mp4`.

## Two Methods

| Method | Speed | Quality | Best for |
|--------|-------|---------|----------|
| **bg-fill** | ~20s | Clean on solid colors | White text on orange/red/blue, solid backgrounds |
| **propainter** | 20min-2h | Natural on textures | Grass, fabric, scenery, complex backgrounds |

**Auto mode** (default) analyzes the watermark region's background complexity:
- Solid/uniform background + bright watermark → bg-fill
- Complex/textured background → ProPainter

Force a specific method:
```bash
python3 scripts/watermark_remover.py VIDEO.mp4 --method bg-fill
python3 scripts/watermark_remover.py VIDEO.mp4 --method propainter
```

## How to Use

When the user asks to remove a watermark from a video:

1. Run the script: `python3 scripts/watermark_remover.py VIDEO.mp4`
2. If auto-detection fails, use `--roi X Y W H` to specify the region manually
3. Monitor progress for long videos (ProPainter mode)
4. Check the verification output for pixel change percentage

### Before starting ProPainter
- Check for existing processes: `ps aux | grep -i propainter | grep -v grep`
- Never start a new run if one is already active
- ProPainter must run from its install directory (weights are relative paths)

### Options

```
--method auto|bg-fill|propainter   Removal method (default: auto)
--roi X Y W H                      Manual watermark region (skip detection)
--resize-ratio auto|0.5|1.0        ProPainter resolution (default: auto)
-o OUTPUT.mp4                      Output path
--keep-temp                        Keep temp files for debugging
--crf N                            HEVC quality (default: 20)
```

## bg-fill Method

Samples background color from bands above and below the watermark strip
(~50px out), interpolates across the strip per-column, and replaces the
watermark with feathered edges.

Best when background is a solid color or simple gradient. Produces perfectly
uniform results — no pale traces. ~20 seconds for 1400 frames.

Limitation: if the watermark area has complex content behind it (person walking
through, texture details), bg-fill will flatten it to a solid color.

## ProPainter Method

AI video inpainting using optical flow and temporal attention. Crops the
watermark region + 120px padding, processes in 500-frame chunks (to manage
RAFT optical flow computation time), then blends back with feathered mask.

Setup (one-time):
1. Clone: `git clone --depth 1 https://ghfast.top/https://github.com/sczhou/ProPainter.git`
2. Download 3 weight files (~190MB) to `weights/`
3. Patch `model/misc.py` for CPU mode on macOS (MPS crashes on large tensors)
4. On this machine: `/Users/hongzhanwang/ProPainter/`

Key parameters:
- `--resize-ratio 0.5`: 2x faster, fine for complex backgrounds
- `--resize-ratio 1.0`: Full resolution, better for high-contrast watermarks
- Chunking: automatic for >510 frames, 10-frame overlap between chunks

## Detection

Auto-detection uses temporal brightness + variance analysis: a static
watermark is bright in EVERY frame AND barely varies across frames. Works
for white/light watermarks on most backgrounds.

Fails on: extremely faint watermarks (brightness <100), moving watermarks,
watermarks on very bright static backgrounds. Use `--roi` as fallback.

## Verification

Pure Python — no vision model. Compares pixel differences in the watermark
region between original and output at the mid-frame:
```
Watermark region pixel change: 12753 px (96.6%)
✅ Watermark region modified (96.6% of region changed)
```

## Output Format

All outputs are HEVC/H.265 with:
- `-tag:v hvc1` (QuickTime compatible, not hev1)
- bt709 color metadata (color_primaries, color_transfer, colormatrix, range)
- Audio copied from original

## Pitfalls

- **ProPainter must run from its directory.** Weight paths are relative to CWD.
  The script handles this automatically with `cwd=propainter_dir`.
- **PYTHONPATH conflict.** Hermes' utils.py shadows ProPainter's utils/ package.
  Script clears PYTHONPATH when spawning ProPainter.
- **Long videos need chunking.** >500 frames = automatic 500-frame chunks with
  10-frame overlap. RAFT optical flow computation is the bottleneck — it runs
  silently before the progress bar appears (~5min per chunk at ratio 0.5,
  ~15min at ratio 1.0).
- **Check for runaway processes.** ProPainter child can outlive the parent.
  `ps aux | sort -k3 -rn | head -5` to find high-CPU Python processes
  (not `process list` — it only tracks bash wrappers, misses Python children).
  **Never kill a ProPainter process without asking the user first.**
  Before starting a new run, always check for existing processes.
- **bg-fill vs ProPainter tradeoff.** bg-fill is 100x faster and cleaner on
  solid backgrounds, but ProPainter produces more natural transitions and
  preserves texture. When quality matters more than speed, use ProPainter.
  When speed matters or background is solid, use bg-fill.

## Files

- `scripts/watermark_remover.py` — Main script. All-in-one: detect, select
  method, remove, verify. This is the only file you need to run.
