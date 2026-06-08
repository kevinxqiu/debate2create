#!/usr/bin/env python3
"""Compose robot MP4 clips into presentation layouts with ffmpeg."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path


def _font_path() -> str:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ):
        if candidate.exists():
            return str(candidate)
    return ""


def _drawtext_filter(text: str, *, x: str, y: str, size: int, border: int = 8) -> str:
    escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    font = _font_path()
    font_part = f"fontfile='{font}':" if font else ""
    return (
        f"drawtext={font_part}text='{escaped}':"
        f"x={x}:y={y}:fontsize={size}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw={border}"
    )


def _layout_filters(
    count: int,
    columns: int,
    cell_width: int,
    cell_height: int,
    fps: int,
    label_font_size: int,
    labels: list[str],
) -> tuple[str, str]:
    filters: list[str] = []
    streams: list[str] = []
    for idx in range(count):
        chain = (
            f"[{idx}:v]fps={fps},setpts=PTS-STARTPTS,"
            f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
            f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
        if idx < len(labels) and labels[idx]:
            chain += "," + _drawtext_filter(
                labels[idx], x="18", y="18", size=label_font_size
            )
        stream = f"v{idx}"
        filters.append(f"{chain}[{stream}]")
        streams.append(f"[{stream}]")

    rows: list[str] = []
    total_rows = math.ceil(count / columns)
    full_row_width = cell_width * columns
    for row_idx in range(total_rows):
        row_streams = streams[row_idx * columns : (row_idx + 1) * columns]
        if len(row_streams) == 1:
            base_stream = row_streams[0].strip("[]")
        else:
            base_stream = f"rowbase{row_idx}"
            filters.append(
                f"{''.join(row_streams)}hstack=inputs={len(row_streams)}:shortest=1[{base_stream}]"
            )

        if columns > 1 and len(row_streams) < columns:
            out = f"row{row_idx}"
            filters.append(
                f"[{base_stream}]pad={full_row_width}:{cell_height}:0:0:color=black[{out}]"
            )
            rows.append(f"[{out}]")
        else:
            rows.append(f"[{base_stream}]")

    if len(rows) == 1:
        output_stream = rows[0].strip("[]")
    else:
        output_stream = "grid"
        filters.append(
            f"{''.join(rows)}vstack=inputs={len(rows)}:shortest=1[{output_stream}]"
        )

    return ";".join(filters), output_stream


def compose(args: argparse.Namespace) -> None:
    inputs = [Path(path).expanduser().resolve() for path in args.inputs]
    output = Path(args.output).expanduser().resolve()
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"Input video not found: {path}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists, pass --overwrite to replace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = args.columns or len(inputs)
    if columns < 1:
        raise SystemExit("--columns must be at least 1")
    labels = args.labels or []
    filter_complex, out_stream = _layout_filters(
        len(inputs),
        columns,
        args.cell_width,
        args.cell_height,
        args.fps,
        args.label_font_size,
        labels,
    )
    if args.title:
        titled_stream = "titled"
        filter_complex += (
            ";"
            + f"[{out_stream}]"
            + _drawtext_filter(
                args.title,
                x="(w-text_w)/2",
                y="h-text_h-24",
                size=args.title_font_size,
                border=10,
            )
            + f"[{titled_stream}]"
        )
        out_stream = titled_stream

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if args.overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    for path in inputs:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{out_stream}]",
            "-an",
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(args.crf),
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    subprocess.run(cmd, check=True)
    print(f"Wrote composed MP4: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input MP4 files.")
    parser.add_argument("--output", required=True, help="Output MP4 file.")
    parser.add_argument("--labels", nargs="*", default=[], help="Per-input labels.")
    parser.add_argument(
        "--title", help="Optional title label for the whole composition."
    )
    parser.add_argument(
        "--columns",
        type=int,
        help="Number of columns in the grid. Defaults to one row.",
    )
    parser.add_argument("--cell-width", type=int, default=960)
    parser.add_argument("--cell-height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    parser.add_argument("--label-font-size", type=int, default=26)
    parser.add_argument("--title-font-size", type=int, default=32)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    compose(build_parser().parse_args())


if __name__ == "__main__":
    main()
