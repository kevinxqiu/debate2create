#!/usr/bin/env python3
"""Capture a Brax HTML rollout as a deterministic MP4.

The script opens a Brax `html.render(...)` output in headless Chrome, exposes the
viewer object in a temporary patched copy, seeks the animation at fixed times,
screenshots only the canvas element, annotates frames, and encodes H.264 MP4.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


VIEWER_SNIPPETS = (
    "var viewer = new Viewer(domElement, system);",
    "const viewer = new Viewer(domElement, system);",
)


def _patch_html(input_html: Path, output_html: Path) -> None:
    text = input_html.read_text(encoding="utf-8")
    replacement = (
        "window.braxViewer = new Viewer(domElement, system);\n"
        "      window.braxViewerReady = true;"
    )
    for snippet in VIEWER_SNIPPETS:
        if snippet in text:
            output_html.write_text(text.replace(snippet, replacement), encoding="utf-8")
            return
    raise RuntimeError(
        "Could not expose Brax viewer. Expected a generated html.render page "
        "containing 'var viewer = new Viewer(domElement, system);'."
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _annotate_frame(
    frame_path: Path,
    *,
    label: str | None,
    sublabel: str | None,
    frame_idx: int,
    frame_count: int,
    font_size: int,
    progress: bool,
) -> None:
    if not label and not sublabel and not progress:
        return

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(font_size)
    small_font = _load_font(max(14, int(font_size * 0.62)))
    pad = max(12, int(font_size * 0.55))

    y = pad
    if label:
        bbox = draw.textbbox((0, 0), label, font=font)
        box_w = bbox[2] - bbox[0] + 2 * pad
        box_h = bbox[3] - bbox[1] + 2 * int(pad * 0.75)
        draw.rounded_rectangle(
            (pad, y, pad + box_w, y + box_h), radius=5, fill=(0, 0, 0, 185)
        )
        draw.text(
            (pad * 2, y + int(pad * 0.55)), label, fill=(255, 255, 255, 255), font=font
        )
        y += box_h + int(pad * 0.45)

    if sublabel:
        bbox = draw.textbbox((0, 0), sublabel, font=small_font)
        box_w = bbox[2] - bbox[0] + 2 * pad
        box_h = bbox[3] - bbox[1] + 2 * int(pad * 0.65)
        draw.rounded_rectangle(
            (pad, y, pad + box_w, y + box_h), radius=5, fill=(0, 0, 0, 150)
        )
        draw.text(
            (pad * 2, y + int(pad * 0.45)),
            sublabel,
            fill=(245, 245, 245, 255),
            font=small_font,
        )

    if progress and frame_count > 1:
        bar_w = image.width - 2 * pad
        bar_h = max(4, int(font_size * 0.14))
        x0 = pad
        y0 = image.height - pad - bar_h
        frac = frame_idx / float(frame_count - 1)
        draw.rectangle((x0, y0, x0 + bar_w, y0 + bar_h), fill=(0, 0, 0, 95))
        draw.rectangle(
            (x0, y0, x0 + int(bar_w * frac), y0 + bar_h), fill=(255, 255, 255, 210)
        )

    image.save(frame_path)


def _encode_mp4(
    frames_dir: Path, output: Path, fps: int, crf: int, overwrite: bool
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def _camera_script() -> str:
    return """
    async ({time, width, height, cameraMode, distance, heightOffset}) => {
      const viewer = window.braxViewer;
      viewer.animator.pause();
      viewer.animator.seek(time);
      viewer.camera.follow = false;
      viewer.camera.freezeAngle = true;
      viewer.setSize(width, height);

      const Vector3 = viewer.camera.position.constructor;
      const target = new Vector3();
      viewer.target.getWorldPosition(target);

      let ox = distance * 0.55;
      let oy = -distance;
      let oz = heightOffset;
      if (cameraMode === "side") {
        ox = 0;
        oy = -distance;
      } else if (cameraMode === "front") {
        ox = distance;
        oy = 0;
      } else if (cameraMode === "top") {
        ox = 0;
        oy = 0;
        oz = distance;
      }

      viewer.controls.target.copy(target);
      viewer.camera.position.set(target.x + ox, target.y + oy, target.z + oz);
      viewer.camera.lookAt(target);
      viewer.camera.updateProjectionMatrix();
      viewer.render();
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return {duration: viewer.animator.duration || 0};
    }
    """


def capture(args: argparse.Namespace) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Playwright is required: python -m pip install playwright"
        ) from exc

    input_html = Path(args.html).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    frames_dir = (
        Path(args.frames_dir).expanduser().resolve()
        if args.frames_dir
        else output.with_suffix("")
    )
    frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"

    if not input_html.exists():
        raise SystemExit(f"HTML file not found: {input_html}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists, pass --overwrite to replace: {output}")
    if frames_dir.exists() and any(frames_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Frames dir is non-empty, pass --overwrite to replace: {frames_dir}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists() and args.overwrite:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = int(round(args.seconds * args.fps))
    if frame_count <= 0:
        raise SystemExit("--seconds and --fps must produce at least one frame")

    with tempfile.TemporaryDirectory(prefix="brax_html_capture_") as tmp:
        patched_html = Path(tmp) / input_html.name
        _patch_html(input_html, patched_html)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel=args.browser_channel,
                headless=True,
                args=["--disable-gpu", "--allow-file-access-from-files"],
            )
            page = browser.new_page(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=1,
            )
            page.goto(patched_html.as_uri(), wait_until="load", timeout=args.timeout_ms)
            page.wait_for_function(
                "window.braxViewerReady === true && document.querySelector('canvas') !== null",
                timeout=args.timeout_ms,
            )
            page.evaluate(
                """({width, height}) => {
                  document.body.style.margin = "0";
                  document.body.style.overflow = "hidden";
                  const viewer = document.getElementById("brax-viewer");
                  viewer.style.width = `${width}px`;
                  viewer.style.height = `${height}px`;
                  const gui = document.querySelector(".lil-gui");
                  if (gui) gui.style.display = "none";
                  window.braxViewer.setSize(width, height);
                }""",
                {"width": args.width, "height": args.height},
            )
            canvas = page.locator("canvas").first

            duration_info = page.evaluate(
                _camera_script(),
                {
                    "time": args.start,
                    "width": args.width,
                    "height": args.height,
                    "cameraMode": args.camera,
                    "distance": args.distance,
                    "heightOffset": args.camera_height,
                },
            )
            animation_duration = float(duration_info.get("duration") or 0.0)

            for idx in range(frame_count):
                t = args.start + idx / float(args.fps)
                if args.loop and animation_duration > 0:
                    t = t % animation_duration
                page.evaluate(
                    _camera_script(),
                    {
                        "time": t,
                        "width": args.width,
                        "height": args.height,
                        "cameraMode": args.camera,
                        "distance": args.distance,
                        "heightOffset": args.camera_height,
                    },
                )
                frame_path = frames_dir / f"frame_{idx:05d}.png"
                canvas.screenshot(path=str(frame_path))
                _annotate_frame(
                    frame_path,
                    label=args.label,
                    sublabel=args.sublabel,
                    frame_idx=idx,
                    frame_count=frame_count,
                    font_size=args.font_size,
                    progress=bool(args.progress),
                )

            browser.close()

    _encode_mp4(frames_dir, output, args.fps, args.crf, args.overwrite)
    print(f"Wrote MP4: {output}")
    print(f"Wrote frames: {frames_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", required=True, help="Input Brax HTML rollout.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--frames-dir", help="Optional frame directory prefix.")
    parser.add_argument(
        "--seconds", type=float, default=8.0, help="Clip duration in seconds."
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in rollout seconds."
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--width", type=int, default=1280, help="Canvas width.")
    parser.add_argument("--height", type=int, default=720, help="Canvas height.")
    parser.add_argument(
        "--camera",
        choices=["three_quarter", "side", "front", "top"],
        default="three_quarter",
    )
    parser.add_argument(
        "--distance", type=float, default=5.0, help="Camera distance from tracked body."
    )
    parser.add_argument(
        "--camera-height", type=float, default=1.4, help="Camera height offset."
    )
    parser.add_argument("--label", help="Primary label burned into frames.")
    parser.add_argument("--sublabel", help="Secondary label burned into frames.")
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument(
        "--progress", action="store_true", help="Draw a small progress bar."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop animation time if clip exceeds rollout duration.",
    )
    parser.add_argument(
        "--crf", type=int, default=18, help="H.264 CRF, lower is higher quality."
    )
    parser.add_argument(
        "--browser-channel", default="chrome", help="Playwright browser channel."
    )
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    capture(build_parser().parse_args())


if __name__ == "__main__":
    main()
