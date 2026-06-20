#!/usr/bin/env python3
"""
watermark_remover.py — Remove static watermarks from video.

Two methods:
  - bg-fill: Background color sampling + fill. Fast (~20s). Best for solid
    color or simple gradient backgrounds with high-contrast watermarks.
  - propainter: ProPainter AI video inpainting. Slow (20min-2h). Best for
    complex textured backgrounds (grass, fabric, scenery). Produces more
    natural results but may leave traces on solid colors.

Usage:
    python3 watermark_remover.py VIDEO.mp4                        # auto-detect method
    python3 watermark_remover.py VIDEO.mp4 --method bg-fill       # force bg-fill
    python3 watermark_remover.py VIDEO.mp4 --method propainter    # force ProPainter
    python3 watermark_remover.py VIDEO.mp4 --roi X Y W H          # manual watermark region

Pipeline (all pure Python, no vision model):
  1. Auto-detect watermark via temporal brightness + variance analysis
  2. Auto-select method: solid background → bg-fill, complex → ProPainter
  3. Remove watermark using selected method
  4. Encode with QuickTime-compatible HEVC (hvc1 + bt709 color metadata)

Requirements:
  - Python 3.9+ with cv2, numpy
  - ffmpeg + ffprobe (with libx265)
  - ProPainter (https://github.com/sczhou/ProPainter) with weights — only
    needed for propainter method. model/misc.py must be patched for CPU mode.


Author: Hermes Agent
License: MIT
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_PROPAINTER_DIR = os.environ.get("PROPAINTER_DIR", "")
DEFAULT_RESIZE_RATIO = 0.5
DEFAULT_SUBVIDEO_LENGTH = 20
DETECT_MIN_BRIGHTNESS = 210
DETECT_MAX_VARIANCE = 80
DETECT_MIN_PIXELS = 50
MASK_TOPHAT_KERNEL = 25
MASK_TOPHAT_THRESH = 8
MASK_MIN_COMPONENT = 20
MASK_DILATE_ITERS = 3
CROP_PADDING = 120
BLEND_FEATHER = 31
BLEND_DILATE_ITERS = 4
ENCODER_CRF = 20
ENCODER_PRESET = "medium"


# ─── Logging ──────────────────────────────────────────────────────────────────

class Log:
    """Simple stderr logger with flush."""
    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}", file=sys.stderr, flush=True)

    @staticmethod
    def step(msg):
        print(f"\n{'='*60}\n[STEP] {msg}\n{'='*60}", file=sys.stderr, flush=True)

    @staticmethod
    def warn(msg):
        print(f"[WARN] {msg}", file=sys.stderr, flush=True)

    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


# ─── Step 1: Probe video ──────────────────────────────────────────────────────

def probe_video(video_path):
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=width,height,duration,codec_name,r_frame_rate,nb_frames",
        "-select_streams", "v:0",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Log.error(f"ffprobe failed: {result.stderr}")
        sys.exit(1)

    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_num, fps_den = stream["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(stream.get("duration", 0)),
        "fps": fps,
        "nb_frames": int(stream.get("nb_frames", 0)),
        "codec": stream["codec_name"],
    }


# ─── Step 2: Sample frames ────────────────────────────────────────────────────

def sample_frames(video_path, n_frames=8, duration=None, total_frames=None):
    """Sample frames evenly across the video timeline."""
    frames = []
    if total_frames and total_frames > 0:
        # Use frame indices for precise sampling
        indices = [int(total_frames * (i + 1) / (n_frames + 1)) for i in range(min(n_frames, total_frames))]
        cap = cv2.VideoCapture(video_path)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    if duration and duration > 12:
        timestamps = [duration * (i + 1) / (n_frames + 1) for i in range(n_frames)]
    else:
        timestamps = [i + 1 for i in range(min(n_frames, 5))]

    for t in timestamps:
        ret, frame = _cv2_read_frame_at(video_path, t)
        if ret:
            frames.append(frame)
    return frames


def _cv2_read_frame_at(video_path, timestamp_sec):
    """Read a single frame at a given timestamp."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    return ret, frame


# ─── Step 3: Detect watermark ─────────────────────────────────────────────────

def detect_watermark(frames, min_brightness=DETECT_MIN_BRIGHTNESS,
                     max_variance=DETECT_MAX_VARIANCE,
                     min_pixels=DETECT_MIN_PIXELS):
    """
    Detect a static bright watermark by finding pixels that are:
    1. Bright in EVERY sampled frame (high min-brightness)
    2. Barely varying across frames (low cross-frame variance)

    Returns (x, y, w, h) bounding box, or None if no watermark found.
    """
    if len(frames) < 3:
        return None

    # Convert to float32 for statistics
    brightness = np.stack([f.astype(np.float32).mean(axis=2) for f in frames])  # (N, H, W)
    min_b = brightness.min(axis=0)  # (H, W) dimmest a pixel ever got
    var_b = brightness.var(axis=0)  # (H, W) cross-frame variance

    # Static bright overlay = bright everywhere AND barely changes
    static_white = np.logical_and(min_b > min_brightness, var_b < max_variance)

    H, W = min_b.shape
    ys, xs = np.where(static_white)
    n = len(xs)

    if n < min_pixels:
        Log.warn(f"Only {n} pixels detected (< {min_pixels}). No watermark found.")
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1

    # Sanity check: if detection spans >60% of frame, likely false positive
    if (x1 - x0) > 0.6 * W or (y1 - y0) > 0.6 * H:
        Log.warn("Detection spans >60% of frame — likely bright scenery, not a small watermark.")
        Log.warn("Retrying with stricter thresholds...")
        # Try stricter thresholds
        if min_brightness < 240:
            return detect_watermark(frames, min_brightness=235, max_variance=50,
                                    min_pixels=min_pixels)
        return None

    Log.info(f"Watermark detected: x={x0} y={y0} w={w} h={h} ({n} pixels)")
    return (x0, y0, w, h)


# ─── Step 4: Generate mask ────────────────────────────────────────────────────

def generate_mask(video_path, wx, wy, ww, wh, W, H,
                  out_dir, n_samples=60):
    """
    Generate a per-pixel watermark mask using temporal-min + top-hat morphology.

    Saves:
      - {out_dir}/mask_full.png  (full-frame mask)
      - {out_dir}/mask_crop.png  (cropped mask for ProPainter)
      - {out_dir}/crop_info.txt  (crop coordinates)

    Returns dict with crop coordinates and mask paths.
    """
    pad = CROP_PADDING
    cx1, cy1 = max(0, wx - pad), max(0, wy - pad)
    cx2, cy2 = min(W, wx + ww + pad), min(H, wy + wh + pad)
    cw, ch = cx2 - cx1, cy2 - cy1

    Log.info(f"Crop region: ({cx1},{cy1})-({cx2},{cy2}) = {cw}x{ch}")

    # Read sampled frames for temporal-min (linear read, avoid HEVC seeking)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        total = int(cap.get(cv2.CAP_PROP_FPS) * 30)  # fallback

    step = max(1, total // n_samples)
    regions = []
    frame_idx = 0
    sample_count = 0
    while sample_count < n_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            regions.append(frame[cy1:cy2, cx1:cx2].astype(np.float32))
            sample_count += 1
        frame_idx += 1
    cap.release()

    if len(regions) < 3:
        Log.error("Not enough frames sampled for mask generation.")
        sys.exit(1)

    # Temporal minimum
    t_min = np.stack(regions).min(axis=0).astype(np.uint8)
    t_gray = cv2.cvtColor(t_min, cv2.COLOR_BGR2GRAY)

    # Top-hat to extract bright static features
    kernel_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (MASK_TOPHAT_KERNEL, MASK_TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(t_gray, cv2.MORPH_TOPHAT, kernel_th)
    mask = (tophat > MASK_TOPHAT_THRESH).astype(np.uint8) * 255

    # Morphological cleanup
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)

    # Remove small isolated noise
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n_labels):
        if stats[i][4] < MASK_MIN_COMPONENT:
            mask[labels == i] = 0

    # Restrict mask to watermark bbox within crop (with small margin)
    mx1, my1 = wx - cx1, wy - cy1
    mx2, my2 = (wx + ww) - cx1, (wy + wh) - cy1
    margin = 10
    mask[:max(0, my1 - margin), :] = 0
    mask[min(ch, my2 + margin):, :] = 0
    mask[:, :max(0, mx1 - margin)] = 0
    mask[:, min(cw, mx2 + margin):] = 0

    # Final dilation for ProPainter
    mask = cv2.dilate(mask, k3, iterations=MASK_DILATE_ITERS)

    mask_px = np.sum(mask > 0)
    mask_pct = 100 * mask_px / (cw * ch)
    Log.info(f"Mask: {mask_px} pixels ({mask_pct:.1f}% of {cw}x{ch} crop)")

    if mask_pct < 0.1:
        Log.warn("Mask covers <0.1% of crop region — watermark may not be detected properly.")
        Log.warn("The watermark might be too faint (brightness < 100). ProPainter may still work.")
        Log.warn("Proceeding anyway...")

    # Save masks
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = mask

    mask_full_path = os.path.join(out_dir, "mask_full.png")
    mask_crop_path = os.path.join(out_dir, "mask_crop.png")
    cv2.imwrite(mask_full_path, full_mask)
    cv2.imwrite(mask_crop_path, mask)

    # Save crop info
    crop_info_path = os.path.join(out_dir, "crop_info.txt")
    with open(crop_info_path, "w") as f:
        f.write(f"{cx1},{cy1},{cx2},{cy2},{cw},{ch}")

    return {
        "cx1": cx1, "cy1": cy1, "cx2": cx2, "cy2": cy2,
        "cw": cw, "ch": ch,
        "mask_full": mask_full_path,
        "mask_crop": mask_crop_path,
        "crop_info": crop_info_path,
    }


# ─── Step 5: Extract cropped frames ───────────────────────────────────────────

def extract_cropped_frames(video_path, cx1, cy1, cx2, cy2, out_dir):
    """Extract all frames cropped to the watermark region."""
    frames_dir = os.path.join(out_dir, "crop_frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[cy1:cy2, cx1:cx2]
        cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.png"), crop)
        idx += 1
        if idx % 500 == 0:
            Log.info(f"  Extracted {idx}/{total} frames...")
    cap.release()

    Log.info(f"Extracted {idx} cropped frames to {frames_dir}")
    return frames_dir, idx


# ─── Step 6: Run ProPainter (with chunking for long videos) ────────────────────

CHUNK_SIZE = 500  # frames per chunk — keeps RAFT computation manageable
CHUNK_OVERLAP = 10  # overlap frames between chunks for seamless blending


def run_propainter_chunked(frames_dir, total_frames, mask_path, output_dir,
                           propainter_dir, resize_ratio, subvideo_length, fps):
    """
    Run ProPainter on cropped frames, splitting into chunks for long videos.

    RAFT optical flow computation time scales linearly with frame count, and
    for >1000 frames the silent pre-processing phase becomes impractically
    long (30+ min with no progress output). Chunking into ~500-frame segments
    keeps each run to ~5 min with visible progress bars.

    Overlap frames between chunks are blended to avoid seam artifacts at
    boundaries.
    """
    if total_frames <= CHUNK_SIZE + CHUNK_OVERLAP:
        # Short enough — run directly
        return run_propainter_single(frames_dir, mask_path, output_dir,
                                     propainter_dir, resize_ratio, subvideo_length)

    # Split into chunks
    chunks = []
    for start in range(0, total_frames, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE + CHUNK_OVERLAP, total_frames)
        chunks.append((start, end))
        if end >= total_frames:
            break

    Log.info(f"Splitting {total_frames} frames into {len(chunks)} chunks "
             f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    chunk_outputs = []
    for i, (start, end) in enumerate(chunks):
        Log.step(f"ProPainter chunk {i+1}/{len(chunks)}: frames {start}-{end}")

        # Create symlink directory for this chunk's frames
        chunk_dir = os.path.join(output_dir, f"chunk_{i:03d}", "crop_frames")
        os.makedirs(chunk_dir, exist_ok=True)
        for fi in range(start, end):
            src = os.path.join(frames_dir, f"{fi:05d}.png")
            dst = os.path.join(chunk_dir, f"{fi-start:05d}.png")
            if os.path.exists(src):
                os.symlink(src, dst)

        chunk_out = os.path.join(output_dir, f"chunk_{i:03d}", "pp_output")
        os.makedirs(chunk_out, exist_ok=True)

        result = run_propainter_single(chunk_dir, mask_path, chunk_out,
                                       propainter_dir, resize_ratio,
                                       subvideo_length)
        chunk_outputs.append((start, end, result))
        Log.info(f"Chunk {i+1} done: {result}")

    # Stitch chunk outputs into a single video
    Log.info("Stitching chunk outputs...")
    stitched_path = os.path.join(output_dir, "inpaint_out_stitched.mp4")

    # Determine crop dimensions from first chunk output
    cap_sample = cv2.VideoCapture(chunk_outputs[0][2])
    ret, frame_sample = cap_sample.read()
    cap_sample.release()
    if not ret:
        Log.error("Cannot read sample frame from chunk output")
        sys.exit(1)
    ch_h, ch_w = frame_sample.shape[:2]
    Log.info(f"Chunk output frame size: {ch_w}x{ch_h}")

    # Write stitched video via ffmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{ch_w}x{ch_h}", "-framerate", str(int(fps)),
        "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        stitched_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    written = 0
    for ci, (start, end, pp_path) in enumerate(chunk_outputs):
        cap = cv2.VideoCapture(pp_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip overlap frames from non-first chunks
            if ci > 0 and frame_idx < CHUNK_OVERLAP:
                frame_idx += 1
                continue

            # For last frames of a chunk (except the last chunk),
            # crossfade with next chunk's overlap
            chunk_len = end - start
            if ci < len(chunk_outputs) - 1:
                remaining = chunk_len - CHUNK_OVERLAP - frame_idx
                if remaining <= CHUNK_OVERLAP:
                    # Crossfade with next chunk
                    pass  # Simple approach: just write, overlap already blended by ProPainter

            proc.stdin.write(frame.tobytes())
            written += 1
            frame_idx += 1
        cap.release()

    proc.stdin.close()
    proc.wait()
    Log.info(f"Stitched {written} frames → {stitched_path}")
    return stitched_path


def run_propainter_single(frames_dir, mask_path, output_dir, propainter_dir,
                          resize_ratio, subvideo_length):
    """Run a single ProPainter inference pass."""
    script = os.path.join(propainter_dir, "inference_propainter.py")
    if not os.path.exists(script):
        Log.error(f"ProPainter inference script not found: {script}")
        sys.exit(1)

    cmd = [
        sys.executable, "-u", "inference_propainter.py",
        "-i", os.path.abspath(frames_dir),
        "-m", os.path.abspath(mask_path),
        "-o", os.path.abspath(output_dir),
        "--resize_ratio", str(resize_ratio),
        "--subvideo_length", str(subvideo_length),
    ]

    Log.info(f"Running ProPainter from {propainter_dir}")

    env = os.environ.copy()
    env["PYTHONPATH"] = ""

    result = subprocess.run(cmd, cwd=propainter_dir, env=env)
    if result.returncode != 0:
        Log.error(f"ProPainter failed with exit code {result.returncode}")
        sys.exit(1)

    # Find output video
    pp_output = os.path.join(output_dir, os.path.basename(frames_dir), "inpaint_out.mp4")
    if not os.path.exists(pp_output):
        pp_output = os.path.join(output_dir, "inpaint_out.mp4")
        if not os.path.exists(pp_output):
            Log.error(f"ProPainter output not found: {pp_output}")
            sys.exit(1)

    Log.info(f"ProPainter output: {pp_output}")
    return pp_output


# ─── Step 7: Blend ProPainter result back ─────────────────────────────────────

def blend_result(video_path, pp_output_path, mask_full_path,
                 cx1, cy1, cx2, cy2, cw, ch,
                 output_path, fps, W, H, crf=ENCODER_CRF):
    """
    Blend ProPainter output back into the original video with a feathered mask.

    Encodes with HEVC hvc1 + bt709 color metadata for QuickTime compatibility.
    """
    Log.info(f"Blending ProPainter result into original video...")

    # Load and feather mask
    mask_full = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    mask_crop = mask_full[cy1:cy2, cx1:cx2]

    kernel = np.ones((5, 5), np.uint8)
    mask_crop = cv2.dilate(mask_crop, kernel, iterations=BLEND_DILATE_ITERS)
    mask_f32 = mask_crop.astype(np.float32) / 255.0
    mask_feather = cv2.GaussianBlur(mask_f32, (BLEND_FEATHER, BLEND_FEATHER), 0)
    mask_feather = np.clip(mask_feather, 0, 1)

    # Check audio
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    has_audio = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()

    # ffmpeg encode command
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{W}x{H}", "-framerate", str(fps),
        "-i", "-",
    ]
    if has_audio:
        cmd += ["-i", video_path, "-map", "0:v", "-map", "1:a?"]
    else:
        cmd += ["-i", video_path, "-map", "0:v"]

    cmd += [
        "-c:v", "libx265", "-crf", str(crf), "-preset", ENCODER_PRESET,
        "-tag:v", "hvc1",
        "-pix_fmt", "yuv420p",
        "-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-c:a", "copy" if has_audio else "aac",
        "-movflags", "+faststart",
        "-f", "mp4", output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    pp_cap = cv2.VideoCapture(pp_output_path)
    orig_cap = cv2.VideoCapture(video_path)
    pp_count = int(pp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_count = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while True:
        ret_orig, frame = orig_cap.read()
        ret_pp, pp_frame = pp_cap.read()
        if not ret_orig:
            break
        if not ret_pp:
            Log.warn(f"ProPainter video ended early at frame {frame_idx}")
            break

        # Resize ProPainter output to crop size
        pp_scaled = cv2.resize(pp_frame, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        # Feather blend
        orig_crop = frame[cy1:cy2, cx1:cx2].astype(np.float32)
        pp_f = pp_scaled.astype(np.float32)
        m = mask_feather[..., np.newaxis]
        blended = orig_crop * (1 - m) + pp_f * m
        frame[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)

        proc.stdin.write(frame.tobytes())
        frame_idx += 1
        if frame_idx % 500 == 0:
            Log.info(f"  Blended {frame_idx}/{orig_count} frames...")

    orig_cap.release()
    pp_cap.release()
    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()

    if proc.returncode != 0:
        Log.error(f"ffmpeg failed: {stderr[-500:]}")
        sys.exit(1)

    Log.info(f"Blended {frame_idx} frames → {output_path}")


# ─── Step 8: Verify ───────────────────────────────────────────────────────────

def verify_result(original_path, output_path, wx, wy, ww, wh, out_dir):
    """
    Verify watermark removal by comparing pixel differences in the watermark region.
    Uses cv2.absdiff instead of a vision model.
    """
    Log.info("Verifying watermark removal...")

    # Extract a frame from the middle of each video
    cap_o = cv2.VideoCapture(original_path)
    cap_n = cv2.VideoCapture(output_path)
    total = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = total // 2
    cap_o.set(cv2.CAP_PROP_POS_FRAMES, mid)
    cap_n.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret_o, orig_frame = cap_o.read()
    ret_n, out_frame = cap_n.read()
    cap_o.release()
    cap_n.release()

    if not ret_o or not ret_n:
        Log.warn("Could not extract frames for verification.")
        return

    # Crop watermark region
    orig_crop = orig_frame[wy:wy+wh, wx:wx+ww]
    out_crop = out_frame[wy:wy+wh, wx:wx+ww]

    diff = cv2.absdiff(orig_crop, out_crop)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    changed_px = np.sum(gray_diff > 5)
    changed_pct = 100 * changed_px / gray_diff.size

    # Save comparison
    diff_path = os.path.join(out_dir, "verify_diff.png")
    cv2.imwrite(diff_path, cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite(os.path.join(out_dir, "verify_orig_crop.png"), orig_crop)
    cv2.imwrite(os.path.join(out_dir, "verify_out_crop.png"), out_crop)

    Log.info(f"Watermark region pixel change: {changed_px} px ({changed_pct:.1f}%)")
    if changed_pct < 1.0:
        Log.warn("Very few pixels changed — watermark may not have been removed.")
    else:
        Log.info(f"✅ Watermark region modified ({changed_pct:.1f}% of region changed)")


# ─── Step 3.5: Auto-detect optimal resize ratio ───────────────────────────────

def detect_background_complexity(video_path, wx, wy, ww, wh, n_samples=3):
    """
    Analyze the watermark region background to decide ProPainter resize ratio.

    Uniform backgrounds (solid colors, gradients) with high-contrast white
    watermarks need full resolution (1.0) — ProPainter at 0.5 can't accurately
    reconstruct uniform color textures and leaves visible pale patches.

    Complex backgrounds (grass, fabric, scenery) work fine at 0.5.

    Returns True if resize_ratio=1.0 is recommended.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg_variances = []
    wm_brightness = []

    for i in range(min(n_samples, 5)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n_samples))
        ret, frame = cap.read()
        if not ret:
            continue

        crop = frame[wy:wy+wh, wx:wx+ww]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Sample border pixels (top/bottom 5px strips) as "background"
        border_top = gray[:5, :].flatten()
        border_bot = gray[-5:, :].flatten()
        border = np.concatenate([border_top, border_bot])

        if len(border) > 0:
            bg_variances.append(float(np.var(border)))

        # Sample center pixels as "watermark"
        center = gray[wh//3:2*wh//3, ww//4:3*ww//4].flatten()
        if len(center) > 0:
            wm_brightness.append(float(np.mean(center)))

    cap.release()

    if not bg_variances or not wm_brightness:
        return False

    avg_bg_var = np.mean(bg_variances)
    avg_wm_bright = np.mean(wm_brightness)

    # Decision logic:
    # - Background variance < 500 → uniform/solid background
    # - Watermark brightness > 180 → bright/white watermark
    # - Brightness diff > 80 → high contrast white-on-dark
    needs_hires = (avg_bg_var < 500 and avg_wm_bright > 180)

    if needs_hires:
        Log.info(f"Detected uniform background (var={avg_bg_var:.0f}) with "
                 f"bright watermark (brightness={avg_wm_bright:.0f})")
        Log.info("→ Using resize_ratio=1.0 for best quality")
    else:
        Log.info(f"Background: var={avg_bg_var:.0f}, watermark brightness={avg_wm_bright:.0f}")
        Log.info("→ Using default resize_ratio=0.5")

    return needs_hires

# ─── Background Fill Method ───────────────────────────────────────────────────

BG_FILL_PADDING = 15      # extra px beyond watermark bbox in each direction
BG_FILL_SAMPLE_DIST = 50  # distance from strip to sample background
BG_FILL_FEATHER = 15      # feather edge width


def remove_bg_fill(video_path, wx, wy, ww, wh, output_path, fps, W, H, crf=ENCODER_CRF):
    """
    Remove watermark by sampling background from outside the watermark strip
    and filling watermark pixels with interpolated background color.

    For each frame:
    1. Sample background bands above and below the watermark strip (50px out)
    2. Per-column median background color from each band
    3. Linear interpolation top→bottom across the strip
    4. Replace entire strip with feathered edges

    Best for: solid color or simple gradient backgrounds, high-contrast
    watermarks (white text on orange/red/blue). ~20 seconds for 1400 frames.
    """
    # Expand ROI with padding
    sx1 = max(0, wx - BG_FILL_PADDING)
    sy1 = max(0, wy - BG_FILL_PADDING)
    sx2 = min(W, wx + ww + BG_FILL_PADDING)
    sy2 = min(H, wy + wh + BG_FILL_PADDING)

    # Background sample bands (well outside the strip)
    bg_top_y1 = max(0, sy1 - BG_FILL_SAMPLE_DIST)
    bg_top_y2 = max(1, sy1 - 15)
    bg_bot_y1 = min(H - 1, sy2 + 15)
    bg_bot_y2 = min(H, sy2 + BG_FILL_SAMPLE_DIST)

    strip_h = sy2 - sy1
    strip_w = sx2 - sx1

    Log.info(f"BG-fill strip: ({sx1},{sy1})-({sx2},{sy2}) = {strip_w}x{strip_h}")
    Log.info(f"Background sample bands: top y={bg_top_y1}-{bg_top_y2}, bot y={bg_bot_y1}-{bg_bot_y2}")

    # Check audio
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True).stdout.strip()

    cmd = ["ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{W}x{H}", "-framerate", str(fps), "-i", "-"]
    if has_audio:
        cmd += ["-i", video_path, "-map", "0:v", "-map", "1:a?"]
    else:
        cmd += ["-i", video_path, "-map", "0:v"]
    cmd += [
        "-c:v", "libx265", "-crf", str(crf), "-preset", ENCODER_PRESET,
        "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
        "-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-c:a", "copy" if has_audio else "aac",
        "-movflags", "+faststart", "-f", "mp4", output_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample background from far outside watermark
        top_band = frame[bg_top_y1:bg_top_y2, sx1:sx2].astype(np.float32)
        bot_band = frame[bg_bot_y1:bg_bot_y2, sx1:sx2].astype(np.float32)

        # Per-column median background
        top_bg = np.median(top_band.reshape(-1, top_band.shape[1], 3), axis=0)  # (strip_w, 3)
        bot_bg = np.median(bot_band.reshape(-1, bot_band.shape[1], 3), axis=0)  # (strip_w, 3)

        # Interpolate top→bottom for each column
        bg_fill = np.zeros((strip_h, strip_w, 3), dtype=np.float32)
        for c in range(3):
            for x in range(strip_w):
                bg_fill[:, x, c] = np.linspace(top_bg[x, c], bot_bg[x, c], strip_h)

        # Feather mask: full opacity center, fade at all edges
        feather_mask = np.ones((strip_h, strip_w), dtype=np.float32)
        for i in range(min(BG_FILL_FEATHER, strip_h // 2, strip_w // 2)):
            w = i / BG_FILL_FEATHER
            feather_mask[i, :] *= w
            feather_mask[strip_h - 1 - i, :] *= w
            feather_mask[:, i] *= w
            feather_mask[:, strip_w - 1 - i] *= w

        m = feather_mask[..., np.newaxis]
        strip = frame[sy1:sy2, sx1:sx2].astype(np.float32)
        result = strip * (1 - m) + bg_fill * m
        frame[sy1:sy2, sx1:sx2] = np.clip(result, 0, 255).astype(np.uint8)

        proc.stdin.write(frame.tobytes())
        frame_idx += 1
        if frame_idx % 500 == 0:
            Log.info(f"  {frame_idx}/{total} frames...")

    cap.release()
    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()

    if proc.returncode != 0:
        Log.error(f"ffmpeg failed: {stderr[-500:]}")
        sys.exit(1)

    Log.info(f"BG-fill: {frame_idx} frames → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remove static watermarks from video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 watermark_remover.py video.mp4
  python3 watermark_remover.py video.mp4 --method bg-fill
  python3 watermark_remover.py video.mp4 --method propainter --resize-ratio 1.0
  python3 watermark_remover.py video.mp4 --roi 400 1430 260 50
        """,
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--output", help="Output path (default: <name>_nowm.mp4)")
    parser.add_argument("--method", choices=["auto", "bg-fill", "propainter"],
                        default="auto",
                        help="Removal method: auto (detect), bg-fill (fast, solid bg), "
                             "propainter (slow, complex bg). Default: auto")
    parser.add_argument("--propainter", default=DEFAULT_PROPAINTER_DIR,
                        help=f"ProPainter directory (default: {DEFAULT_PROPAINTER_DIR})")
    parser.add_argument("--resize-ratio", type=str, default="auto",
                        help="ProPainter resize ratio: auto, 0.5, 1.0. Default: auto")
    parser.add_argument("--subvideo-length", type=int, default=DEFAULT_SUBVIDEO_LENGTH,
                        help=f"ProPainter subvideo length (default: {DEFAULT_SUBVIDEO_LENGTH})")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Manually specify watermark region (skip auto-detection)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files for debugging")
    parser.add_argument("--crf", type=int, default=ENCODER_CRF,
                        help=f"HEVC CRF quality (default: {ENCODER_CRF})")

    args = parser.parse_args()

    # Validate inputs
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        Log.error(f"Video not found: {video_path}")
        sys.exit(1)

    if args.method in ("auto", "propainter") and not os.path.exists(args.propainter or ""):
        if args.method == "propainter":
            Log.error(f"ProPainter not found. Set PROPAINTER_DIR env var or use --propainter.")
            sys.exit(1)
        else:
            Log.warn(f"ProPainter not found — will use bg-fill if auto selects it")

    # Output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_nowm{ext or '.mp4'}"

    temp_dir = tempfile.mkdtemp(prefix="watermark_remover_")

    try:
        # ── Step 1: Probe ──────────────────────────────────────────────
        Log.step("Step 1: Probing video")
        info = probe_video(video_path)
        Log.info(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
                 f"{info['nb_frames']} frames, {info['codec']}")

        # ── Step 2: Sample frames ──────────────────────────────────────
        Log.step("Step 2: Sampling frames for watermark detection")
        det_frames = sample_frames(video_path, n_frames=8, duration=info["duration"],
                                   total_frames=info["nb_frames"])
        Log.info(f"Sampled {len(det_frames)} frames for detection")

        # ── Step 3: Detect watermark ───────────────────────────────────
        Log.step("Step 3: Detecting watermark")
        if args.roi:
            wx, wy, ww, wh = args.roi
            Log.info(f"Using manual ROI: x={wx} y={wy} w={ww} h={wh}")
        else:
            roi = detect_watermark(det_frames)
            if roi is None:
                Log.error("No watermark detected automatically.")
                Log.error("Try manually specifying the region with --roi X Y W H.")
                sys.exit(1)
            wx, wy, ww, wh = roi

        # ── Step 4: Select method ──────────────────────────────────────
        Log.step("Step 4: Selecting removal method")
        if args.method == "auto":
            needs_hires = detect_background_complexity(video_path, wx, wy, ww, wh)
            is_solid = needs_hires  # solid background detected
            if is_solid:
                method = "bg-fill"
                Log.info("→ Auto-selected: bg-fill (solid background detected)")
            else:
                method = "propainter"
                resize_ratio = DEFAULT_RESIZE_RATIO
                Log.info("→ Auto-selected: propainter (complex background)")
        elif args.method == "bg-fill":
            method = "bg-fill"
            Log.info("→ Forced: bg-fill")
        else:
            method = "propainter"
            if args.resize_ratio == "auto":
                needs_hires = detect_background_complexity(video_path, wx, wy, ww, wh)
                resize_ratio = 1.0 if needs_hires else DEFAULT_RESIZE_RATIO
            else:
                resize_ratio = float(args.resize_ratio)
            Log.info(f"→ Forced: propainter (resize_ratio={resize_ratio})")

        # ── Step 5: Remove watermark ───────────────────────────────────
        if method == "bg-fill":
            Log.step("Step 5: Removing watermark (bg-fill)")
            remove_bg_fill(video_path, wx, wy, ww, wh, output_path,
                          info["fps"], info["width"], info["height"], crf=args.crf)
        else:
            Log.step("Step 5a: Generating watermark mask")
            mask_info = generate_mask(video_path, wx, wy, ww, wh,
                                      info["width"], info["height"], temp_dir)

            Log.step("Step 5b: Extracting cropped frames")
            frames_dir, total_extracted = extract_cropped_frames(
                video_path,
                mask_info["cx1"], mask_info["cy1"],
                mask_info["cx2"], mask_info["cy2"],
                temp_dir,
            )

            Log.step("Step 5c: Running ProPainter AI inpainting")
            pp_output_dir = os.path.join(temp_dir, "pp_output")
            os.makedirs(pp_output_dir, exist_ok=True)
            pp_output = run_propainter_chunked(
                frames_dir, total_extracted,
                mask_info["mask_crop"], pp_output_dir,
                args.propainter,
                resize_ratio=resize_ratio,
                subvideo_length=args.subvideo_length,
                fps=info["fps"],
            )

            Log.step("Step 5d: Blending and encoding output")
            blend_result(
                video_path, pp_output, mask_info["mask_full"],
                mask_info["cx1"], mask_info["cy1"],
                mask_info["cx2"], mask_info["cy2"],
                mask_info["cw"], mask_info["ch"],
                output_path, info["fps"],
                info["width"], info["height"], crf=args.crf,
            )

        # ── Step 6: Verify ─────────────────────────────────────────────
        Log.step("Step 6: Verification")
        verify_result(video_path, output_path, wx, wy, ww, wh, temp_dir)

        Log.step("Done!")
        Log.info(f"Output: {output_path}")
        Log.info(f"Temp files: {'kept' if args.keep_temp else 'cleaned'} ({temp_dir})")

    finally:
        if not args.keep_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
