from __future__ import annotations

from pathlib import Path
from typing import List

from audio_analysis import AudioAnalysisConfig, detect_strong_hits
from video_edit import build_beat_synced_video

# 直接在代码里配置输入/输出，运行 main.py 时无需再传命令行参数。
BASE_DIR = Path(__file__).resolve().parent
AUDIO_PATH = BASE_DIR / "002.mp3"
VIDEO_DIR = BASE_DIR / "voice_extracted"
OUTPUT_PATH = BASE_DIR / "result.mp4"


def collect_video_paths() -> List[str]:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"未找到视频目录：{VIDEO_DIR}")

    videos = sorted(
        str(p) for p in VIDEO_DIR.glob("clip_*.mp4") if p.is_file()
    )
    if not videos:
        raise FileNotFoundError(f"{VIDEO_DIR} 中没有匹配的 clip_*.mp4 文件。")
    return videos


def main() -> None:
    audio_path: str = str(AUDIO_PATH)
    video_paths: List[str] = collect_video_paths()
    output_path: str = str(OUTPUT_PATH)

    print("=== 步骤 1：检测音频重击点 ===")
    config = AudioAnalysisConfig(
        strict_factor=3.0,
        min_hit_interval=0.6,
    )
    hit_times = detect_strong_hits(audio_path, config=config)

    print(f"检测到 {len(hit_times)} 个重击点。")
    if len(hit_times) > 0:
        preview = ", ".join(f"{t:.2f}s" for t in hit_times[:10])
        print(f"前几个重击时间点: {preview}")
        if len(hit_times) > 10:
            print("...")

    print("\n=== 步骤 2：按重击时间拼接视频 ===")
    build_beat_synced_video(
        audio_path=audio_path,
        video_paths=video_paths,
        hit_times=hit_times,
        output_path=output_path,
        fps=25,
    )

    print("\n完成！输出文件：", output_path)


if __name__ == "__main__":
    main()
