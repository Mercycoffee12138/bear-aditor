"""
批量从视频中提取人声并重新合成视频。

需求：
- 输入目录：video_processed
- 输出目录：voice_extracted
- 临时目录：temp_voice_separation
- 处理逻辑：对每个视频抽取音频 -> 用 Spleeter 分离人声 -> 将人声轨重新绑定到视频 -> 输出
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from moviepy.editor import AudioFileClip, VideoFileClip
from spleeter.separator import Separator

# 这些路径直接写死在脚本里，运行时无需额外命令行参数。
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "video" / "sad"
OUTPUT_DIR = BASE_DIR / "voice_extracted"
TEMP_DIR = BASE_DIR / "temp_voice_separation"
TEMP_AUDIO_DIR = TEMP_DIR / "audio"
TEMP_STEM_DIR = TEMP_DIR / "stems"

# 支持的常见视频后缀，可按需自行扩展。
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def list_video_files() -> List[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在：{INPUT_DIR}")
    return sorted(
        [
            p
            for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda p: p.name,
    )


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_STEM_DIR.mkdir(parents=True, exist_ok=True)


def extract_audio_track(video_path: Path, audio_output: Path) -> None:
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise ValueError("该视频没有可用的音频轨道。")
        clip.audio.write_audiofile(
            str(audio_output),
            fps=44100,
            codec="pcm_s16le",
            logger=None,
        )


def run_spleeter(audio_path: Path, separator: Separator) -> Path:
    stem_target = TEMP_STEM_DIR / audio_path.stem
    if stem_target.exists():
        shutil.rmtree(stem_target)
    separator.separate_to_file(str(audio_path), str(TEMP_STEM_DIR))
    vocal_path = stem_target / "vocals.wav"
    if not vocal_path.exists():
        raise FileNotFoundError(f"未找到人声文件：{vocal_path}")
    return vocal_path


def rebuild_video_with_vocals(video_path: Path, vocals_path: Path, output_path: Path) -> None:
    with VideoFileClip(str(video_path)) as video_clip:
        with AudioFileClip(str(vocals_path)) as vocals_clip:
            final_clip = video_clip.set_audio(vocals_clip)
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
            )
            final_clip.close()


def process_video(video_path: Path, separator: Separator) -> None:
    print(f"\n=== 处理视频：{video_path.name} ===")
    temp_audio = TEMP_AUDIO_DIR / f"{video_path.stem}.wav"
    try:
        print("1) 提取音频...")
        extract_audio_track(video_path, temp_audio)

        print("2) 使用 Spleeter 分离人声...")
        vocals_path = run_spleeter(temp_audio, separator)

        output_path = OUTPUT_DIR / video_path.name
        print(f"3) 合成视频 + 人声 -> {output_path}")
        rebuild_video_with_vocals(video_path, vocals_path, output_path)
    finally:
        if temp_audio.exists():
            temp_audio.unlink(missing_ok=True)


def main() -> None:
    ensure_directories()
    video_files = list_video_files()
    if not video_files:
        print(f"在 {INPUT_DIR} 未找到可处理的视频文件。")
        return

    print(f"共检测到 {len(video_files)} 个视频，开始批处理...")
    separator = Separator("spleeter:2stems")

    processed = 0
    failures: List[str] = []

    for video_path in video_files:
        try:
            process_video(video_path, separator)
            processed += 1
        except Exception as exc:
            failures.append(f"{video_path.name}: {exc}")
            print(f"[!] 处理 {video_path.name} 时出错：{exc}")

    print("\n=== 处理完成 ===")
    print(f"成功：{processed} 个视频；失败：{len(failures)} 个。")
    if failures:
        print("以下文件处理失败，请手动检查：")
        for msg in failures:
            print(" -", msg)


if __name__ == "__main__":
    main()
