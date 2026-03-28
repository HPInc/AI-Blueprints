import os
import pyttsx3
import textwrap
import subprocess
import shutil
import soundfile as sf
import numpy as np
from typing import Any, Dict
from pathlib import Path


def generate_test_audio(media_path: Path) -> None:
    """
    Generate a test audio file if it doesn't exist.
    This function uses pyttsx3 to create a synthetic audio file.
    """
    if not media_path.exists():
        tts = pyttsx3.init()  # offline, uses system voices
        tts.setProperty("rate", 120)  # speaking speed
        text = (
            "Hello and welcome to the Agentic Audio RAG demo."
            "This short clip will be chunked and analysed by the workflow."
            "Feel free to ask any question about its content."
        )
        tts.save_to_file(text, str(media_path / "sample_tts.mp3"))
        tts.runAndWait()
        tts.stop()
        print("🎙️  Generated synthetic audio →", media_path)
    else:
        print("🎙️  Using existing file →", media_path)


def generate_meeting_audio(media_path: Path) -> None:
    """Generate a richer meeting audio as a master WAV file for testing."""

    SAMPLE_BASE = media_path / "sample_meeting"
    SAMPLE_BASE.parent.mkdir(parents=True, exist_ok=True)
    MASTER_WAV = SAMPLE_BASE.with_suffix(".wav")

    script = textwrap.dedent("""
        Hello everyone, and welcome to the Project Helios kickoff.
        I'm Alex, the product lead. Also joining: Ben from engineering, Carla from design, and Diego from data.
        Our goal is to ship a private beta by August fifth, focused on the voice search experience.

        Key findings from last week’s user interviews:
        One: people expect sub-second responses for voice queries.
        Two: transcripts must show timestamps and be searchable across meetings.
        Three: mobile users care most about hands-free playback of relevant segments.

        Decisions today:
        We’ll keep the retrieval stack audio-native using CLAP, and let the agent listen with Qwen Omni.
        No full transcription is required for the MVP—only timestamped evidence.
        We will add a reranker to improve precision and avoid duplicate segments.

        Action items:
        Ben will implement the CLAP index with thirty-second windows, due July thirtieth.
        Carla will design the evidence panel that shows start and end timestamps, due August first.
        Diego will instrument latency metrics and a cache hit rate dashboard, due August fifth.

        Constraints and metrics:
        P ninety latency for a query should be under one and a half seconds.
        Minimum five relevant segments returned per query, with a confidence score above point two.
        Cache hit rate target is at least thirty percent for repeated questions.

        If you need help, email helios at example dot com.
        Our next review is on Monday at ten A M central.

        That’s the plan—ask me anything about this meeting.
    """).strip()

    if not MASTER_WAV.exists():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # speaking speed
        # Optional voice selection:
        # for v in engine.getProperty("voices"): print(v.id)
        # engine.setProperty("voice", "english-us")
        print("🎙️  Generating master WAV:", MASTER_WAV)
        engine.save_to_file(script, str(MASTER_WAV))
        engine.runAndWait()
        engine.stop()
    else:
        print("🎙️  Using existing:", MASTER_WAV)
    return MASTER_WAV, SAMPLE_BASE


######### Helper functions for audio conversion ##########


def ensure_ffmpeg_bin() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        binpath = imageio_ffmpeg.get_ffmpeg_exe()
        # Make sure downstream libs can find it
        os.environ["IMAGEIO_FFMPEG_EXE"] = binpath
        os.environ["FFMPEG_BINARY"] = binpath  # used by moviepy if ever needed
        # Prepend its folder to PATH so subprocess can call "ffmpeg"
        ffdir = os.path.dirname(binpath)
        os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")
        return binpath
    except Exception:
        return None


FFMPEG_BIN = ensure_ffmpeg_bin()


def ffmpeg_ok():
    return FFMPEG_BIN is not None or shutil.which("ffmpeg") is not None


def _ffmpeg_cmd() -> list[str]:
    # Use the explicit binary if we found one; else rely on PATH
    exe = FFMPEG_BIN if FFMPEG_BIN else "ffmpeg"
    return [exe, "-y", "-hide_banner", "-loglevel", "error"]


def convert_ffmpeg(src, dst, acodec=None, vcodec=None, extra=None):
    if not ffmpeg_ok():
        print("⚠️  ffmpeg not found; skipping", Path(dst).name)
        return
    cmd = _ffmpeg_cmd() + ["-i", str(src)]
    if vcodec:
        cmd += ["-c:v", vcodec, "-pix_fmt", "yuv420p"]
    if acodec:
        cmd += ["-c:a", acodec]
    if extra:
        cmd += extra
    cmd += [str(dst)]
    try:
        subprocess.run(cmd, check=True)
        print("✅", dst)
    except Exception as e:
        print("⚠️  ffmpeg failed:", dst, "-", e)


def make_video_with_audio_ffmpeg(
    audio_path: Path,
    out_path: Path,
    *,
    size="1280x720",
    fps=24,
    bg_color="black",
    vcodec="libx264",
    acodec="aac",
    extra=None,
):
    """
    Create a solid-color video (lavfi color) and mux the given audio.
    """
    if not ffmpeg_ok():
        print("⚠️  ffmpeg not found; skipping", out_path.name)
        return
    cmd = _ffmpeg_cmd() + [
        # video source: solid color frames
        "-f",
        "lavfi",
        "-r",
        str(fps),
        "-i",
        f"color=c={bg_color}:s={size}",
        # audio source
        "-i",
        str(audio_path),
        "-shortest",
        "-c:v",
        vcodec,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        acodec,
    ]
    if extra:
        cmd += extra
    cmd += [str(out_path)]
    try:
        subprocess.run(cmd, check=True)
        print("✅", out_path)
    except Exception as e:
        print("⚠️  ffmpeg failed:", out_path, "-", e)


######## Convert all supported audio + video formats ########


def generate_and_convert_formats(media_path: Path) -> Dict[str, str]:
    """Convert to all supported audio + video containers (ffmpeg-only, no moviepy)"""

    MASTER_WAV, SAMPLE_BASE = generate_meeting_audio(media_path)

    AUDIO_TARGETS = {
        "wav": MASTER_WAV,
        "flac": SAMPLE_BASE.with_suffix(".flac"),
        "ogg": SAMPLE_BASE.with_suffix(".ogg"),  # Vorbis
        "mp3": SAMPLE_BASE.with_suffix(".mp3"),
        "m4a": SAMPLE_BASE.with_suffix(".m4a"),  # AAC
    }

    # Read the master once for libsndfile outputs
    data, sr = sf.read(MASTER_WAV)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # WAV already exists (MASTER_WAV). FLAC/OGG via soundfile, fallback to ffmpeg if needed.
    for fmt, path in [("flac", AUDIO_TARGETS["flac"]), ("ogg", AUDIO_TARGETS["ogg"])]:
        if not path.exists():
            try:
                if fmt == "flac":
                    sf.write(str(path), data, sr, format="FLAC")
                else:
                    sf.write(str(path), data, sr, format="OGG", subtype="VORBIS")
                print("✅", path)
            except Exception as e:
                print(f"⚠️  soundfile {fmt} failed, trying ffmpeg:", e)
                acodec = "flac" if fmt == "flac" else "libvorbis"
                convert_ffmpeg(MASTER_WAV, path, acodec=acodec)

    # MP3/M4A via ffmpeg
    if ffmpeg_ok():
        if not AUDIO_TARGETS["mp3"].exists():
            convert_ffmpeg(
                MASTER_WAV,
                AUDIO_TARGETS["mp3"],
                acodec="libmp3lame",
                extra=["-b:a", "160k"],
            )
        if not AUDIO_TARGETS["m4a"].exists():
            convert_ffmpeg(
                MASTER_WAV, AUDIO_TARGETS["m4a"], acodec="aac", extra=["-b:a", "160k"]
            )
    else:
        print("⚠️  ffmpeg not found; skipping MP3/M4A.")

    # Video containers via ffmpeg color source + audio
    VIDEO_TARGETS = {
        "mp4": SAMPLE_BASE.with_suffix(".mp4"),
        "mov": SAMPLE_BASE.with_suffix(".mov"),
        "mkv": SAMPLE_BASE.with_suffix(".mkv"),
        "avi": SAMPLE_BASE.with_suffix(".avi"),
        "webm": SAMPLE_BASE.with_suffix(".webm"),
    }

    # H.264 + AAC for MP4/MOV/MKV
    for k in ["mp4", "mov", "mkv"]:
        if not VIDEO_TARGETS[k].exists():
            make_video_with_audio_ffmpeg(
                MASTER_WAV,
                VIDEO_TARGETS[k],
                vcodec="libx264",
                acodec="aac",
                size="1280x720",
                fps=24,
                bg_color="black",
            )

    # AVI prefers MPEG-4 Part 2 + MP3
    if not VIDEO_TARGETS["avi"].exists():
        make_video_with_audio_ffmpeg(
            MASTER_WAV,
            VIDEO_TARGETS["avi"],
            vcodec="mpeg4",
            acodec="libmp3lame",
            size="1280x720",
            fps=24,
            bg_color="black",
            extra=["-qscale:v", "3"],
        )  # quality knob for mpeg4

    # WEBM (VP9 + Opus)
    if not VIDEO_TARGETS["webm"].exists():
        make_video_with_audio_ffmpeg(
            MASTER_WAV,
            VIDEO_TARGETS["webm"],
            vcodec="libvpx-vp9",
            acodec="libopus",
            size="1280x720",
            fps=24,
            bg_color="black",
            extra=["-b:v", "0", "-crf", "32"],
        )

    # --- Summary of generated media ---
    created = {
        k: str(p) for k, p in {**AUDIO_TARGETS, **VIDEO_TARGETS}.items() if p.exists()
    }
    print("Created media:")
    for k, p in created.items():
        print(f"  {k:>4}: {p}")

    return created
