#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_timestamp(value: str) -> float:
    raw = value.strip()
    if not raw:
        raise ValueError("empty timestamp")

    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid timestamp '{value}'")

    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            seconds = minutes * 60 + seconds
        else:
            hours = int(parts[0])
            minutes = int(parts[1])
            secs = float(parts[2])
            seconds = hours * 3600 + minutes * 60 + secs
    except ValueError as error:
        raise ValueError(f"invalid timestamp '{value}'") from error

    if seconds < 0:
        raise ValueError(f"timestamp must be non-negative: '{value}'")
    return seconds


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    whole = int(seconds)
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def slug_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(":", "-")


def read_timestep_markers(path: Path) -> list[float]:
    markers: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            cleaned = re.split(r"\s+#", cleaned, maxsplit=1)[0].strip()
            marker = parse_timestamp(cleaned)
            markers.append(marker)
            if len(markers) > 1 and marker <= markers[-2]:
                previous = format_timestamp(markers[-2])
                current = format_timestamp(marker)
                raise ValueError(
                    f"timesteps must be strictly increasing: line {line_number} "
                    f"('{current}') is not after previous '{previous}'"
                )

    if not markers:
        raise ValueError(f"no timesteps found in '{path}'")
    return markers


def ffprobe_duration_seconds(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed: {stderr}")

    output = result.stdout.strip()
    try:
        return float(output)
    except ValueError as error:
        raise RuntimeError(f"could not parse duration from ffprobe output: '{output}'") from error


def build_segments(markers: list[float], duration: float, include_tail: bool) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    if len(markers) >= 2:
        segments.extend((markers[i], markers[i + 1]) for i in range(len(markers) - 1))

    if include_tail:
        last = markers[-1]
        if duration > last:
            segments.append((last, duration))

    if not segments:
        raise ValueError(
            "not enough timesteps to build clips. Provide at least two markers or use --include-tail with "
            "a marker before video end."
        )

    return segments


def clip_filename(index: int, start: float, end: float, is_tail: bool) -> str:
    start_slug = slug_timestamp(start)
    if is_tail:
        return f"clip_{index:02d}_{start_slug}_to_end.mp4"
    end_slug = slug_timestamp(end)
    return f"clip_{index:02d}_{start_slug}_to_{end_slug}.mp4"


def build_ffmpeg_command(
    input_video: Path,
    output_video: Path,
    start: float,
    end: float,
    overwrite: bool,
    quality_crf: int,
    preset: str,
) -> list[str]:
    time_args = ["-ss", f"{start:.3f}"]
    if end > start:
        time_args += ["-to", f"{end:.3f}"]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        *time_args,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(quality_crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    return command


def ensure_binaries() -> None:
    missing = [binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"required binaries not found in PATH: {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a video into snippets using timestep markers. Markers in the timesteps file are treated "
            "as clip boundaries: each marker -> next marker. Optionally include tail (last marker -> end)."
        )
    )
    parser.add_argument("video_path", type=Path, help="Path to input video")
    parser.add_argument(
        "--timesteps",
        type=Path,
        default=Path("video-timesteps.txt"),
        help="Path to timestep file (default: ./video-timesteps.txt)",
    )
    parser.add_argument(
        "--output-folder-name",
        default=None,
        help="Name of output folder created in the same directory as the input video (default: <video_stem>_snippets)",
    )
    parser.add_argument(
        "--include-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include final clip from last marker to end of video (default: enabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output clip files",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality CRF for snippets (lower is higher quality, default: 18)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 preset (ultrafast..veryslow, default: medium)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned clips and ffmpeg commands without writing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = args.video_path.expanduser().resolve()
    timesteps_path = args.timesteps.expanduser().resolve()

    if not video_path.exists() or not video_path.is_file():
        print(f"Error: input video not found: {video_path}", file=sys.stderr)
        return 1
    if not timesteps_path.exists() or not timesteps_path.is_file():
        print(f"Error: timesteps file not found: {timesteps_path}", file=sys.stderr)
        return 1

    try:
        ensure_binaries()
        markers = read_timestep_markers(timesteps_path)
        duration = ffprobe_duration_seconds(video_path)
        segments = build_segments(markers, duration, include_tail=args.include_tail)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    folder_name = args.output_folder_name or f"{video_path.stem}_snippets"
    output_dir = video_path.parent / folder_name
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.overwrite:
        for index, (start, end) in enumerate(segments, start=1):
            is_tail = index == len(segments) and end >= duration and args.include_tail
            out_file = output_dir / clip_filename(index, start, end, is_tail)
            if out_file.exists():
                print(
                    f"Error: output file exists ({out_file}). Use --overwrite to replace existing clips.",
                    file=sys.stderr,
                )
                return 1

    print(f"Input video:    {video_path}")
    print(f"Timesteps file: {timesteps_path}")
    print(f"Output folder:  {output_dir}")
    print(f"Video length:   {format_timestamp(duration)}")
    print(f"Planned clips:  {len(segments)}")

    for index, (start, end) in enumerate(segments, start=1):
        is_tail = index == len(segments) and end >= duration and args.include_tail
        output_file = output_dir / clip_filename(index, start, end, is_tail)
        command = build_ffmpeg_command(
            input_video=video_path,
            output_video=output_file,
            start=start,
            end=end,
            overwrite=args.overwrite,
            quality_crf=args.crf,
            preset=args.preset,
        )

        print(f"[{index:02d}/{len(segments):02d}] {format_timestamp(start)} -> {format_timestamp(end)}")
        print(f"       {output_file.name}")

        if args.dry_run:
            print(f"       cmd: {' '.join(command)}")
            continue

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Error: ffmpeg failed for clip {index}: {output_file}", file=sys.stderr)
            return result.returncode

    if args.dry_run:
        print("Dry run complete. No files were written.")
    else:
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
