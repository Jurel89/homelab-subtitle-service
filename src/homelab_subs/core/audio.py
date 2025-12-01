from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe command fails or binaries are missing."""


@dataclass
class VideoInfo:
    path: Path
    duration: Optional[float]  # seconds, None if unknown
    has_audio: bool
    has_video: bool


class FFmpeg:
    """
    Thin, Pythonic wrapper around ffmpeg and ffprobe.

    This class hides all subprocess details and exposes high-level
    methods like `probe_video` and `extract_audio_to_wav`.
    """

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    # ---------- internal helpers ----------

    def _check_binary(self, bin_name: str) -> None:
        if shutil.which(bin_name) is None:
            raise FFmpegError(
                f"Required binary '{bin_name}' not found in PATH.\n"
                "\n"
                "Please install FFmpeg:\n"
                "  macOS:   brew install ffmpeg\n"
                "  Ubuntu:  sudo apt install ffmpeg\n"
                "  Windows: https://ffmpeg.org/download.html\n"
            )


    def ensure_available(self) -> None:
        """
        Ensure ffmpeg and ffprobe are available in PATH.

        Raises FFmpegError if not found.
        """
        self._check_binary(self.ffmpeg_bin)
        self._check_binary(self.ffprobe_bin)

    def _run(self, cmd: list[str]) -> str:
        """
        Run a command and return stdout.

        Raises FFmpegError on failure.
        """
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            raise FFmpegError(
                f"Command failed: {' '.join(cmd)}\n"
                f"Exit code: {exc.returncode}\n"
                f"Stdout: {exc.stdout}\n"
                f"Stderr: {exc.stderr}"
            ) from exc

    # ---------- public API ----------

    def probe_video(self, path: Path) -> VideoInfo:
        """
        Return basic info about a video file using ffprobe.

        Parameters
        ----------
        path : Path
            Path to the input video file.

        Returns
        -------
        VideoInfo

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        FFmpegError
            If ffprobe fails or is not available.
        """
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        self.ensure_available()

        cmd = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        stdout = self._run(cmd)
        data = json.loads(stdout)

        duration: Optional[float] = None
        has_audio = False
        has_video = False

        fmt = data.get("format") or {}
        if "duration" in fmt:
            try:
                duration = float(fmt["duration"])
            except (TypeError, ValueError):
                duration = None

        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type")
            if codec_type == "audio":
                has_audio = True
            elif codec_type == "video":
                has_video = True

        return VideoInfo(
            path=path,
            duration=duration,
            has_audio=has_audio,
            has_video=has_video,
        )

    def extract_audio_to_wav(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: int = 16_000,
        channels: int = 1,
        overwrite: bool = True,
    ) -> Path:
        """
        Extract audio from a video as a WAV file suitable for Whisper.

        Parameters
        ----------
        video_path : Path
            Input video file.
        output_path : Optional[Path]
            Where to write the WAV file. If None, a temp file will be created
            in a temporary directory and its Path returned.
        sample_rate : int
            Target sample rate in Hz (default: 16000).
        channels : int
            Number of audio channels (default: 1 = mono).
        overwrite : bool
            Whether to overwrite an existing output file.

        Returns
        -------
        Path
            Path to the generated WAV file.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        FFmpegError
            If ffmpeg fails or is not available.
        """
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.ensure_available()

        if output_path is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="homelab_subs_audio_"))
            output_path = tmp_dir / f"{video_path.stem}_audio.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
        ]

        # Overwrite / no-overwrite flags
        cmd.append("-y" if overwrite else "-n")

        # Input file
        cmd.extend(["-i", str(video_path)])

        # Audio output settings
        cmd.extend(
            [
                "-vn",              # no video
                "-acodec",
                "pcm_s16le",       # 16-bit little-endian PCM
                "-ar",
                str(sample_rate),  # sample rate
                "-ac",
                str(channels),     # channels
                str(output_path),
            ]
        )

        self._run(cmd)

        if not output_path.is_file():
            raise FFmpegError(
                f"ffmpeg reported success but output file not found: {output_path}"
            )

        return output_path


# A convenient default instance for most of the codebase to use
default_ffmpeg = FFmpeg()
