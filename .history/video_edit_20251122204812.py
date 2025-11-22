# video_edit.py
"""
视频编辑模块：根据重击时间点，将多段素材视频拼接成卡点视频。

本版本与之前的区别：
- 不再用 BGM 完全替换视频原声。
- 而是保留每段素材视频自带的声音，再把 BGM 叠加为背景音乐。

混音策略：
- final_audio = video_original_audio * video_volume + bgm_audio * bgm_volume
"""

from __future__ import annotations

from typing import List

import numpy as np
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeAudioClip,
)
from moviepy.audio.fx.all import audio_normalize


def build_beat_synced_video(
    audio_path: str,
    video_paths: List[str],
    hit_times: np.ndarray,
    output_path: str,
    fps: int = 25,
    bgm_volume: float = 0.2,
    video_volume: float = 1.0,
) -> None:
    """
    按重击时间拼接视频并导出成品（保留原声 + 叠加 BGM）。

    :param audio_path: BGM 音频文件路径
    :param video_paths: 视频素材路径列表
    :param hit_times: 重击时间（秒），升序的一维 numpy 数组
    :param output_path: 输出成片视频文件路径
    :param fps: 输出视频帧率
    :param bgm_volume: BGM 音量系数（0~1 左右，默认 0.4）
    :param video_volume: 视频原声音量系数（默认 1.0）
    """
    if not video_paths:
        raise ValueError("video_paths 为空，请至少提供一个视频素材。")

    # 1. 加载 BGM，并先做响度归一化，只用于确定时长，真正混音时再重新裁剪
    bgm_full = AudioFileClip(audio_path)
    bgm_full = bgm_full.fx(audio_normalize)
    audio_duration = bgm_full.duration

    # 2. 清理 & 边界处理重击时间
    hit_times = np.array(hit_times, dtype=float)
    hit_times = hit_times[(hit_times > 0.0) & (hit_times < audio_duration)]
    hit_times = np.unique(np.round(hit_times, 3))  # 去重 & 保留三位小数

    # 划分 BGM 时间片段
    if hit_times.size == 0:
        segment_starts = [0.0]
        segment_ends = [audio_duration]
    else:
        segment_starts = [0.0] + hit_times.tolist()
        segment_ends = hit_times.tolist() + [audio_duration]

    # 3. 加载视频素材（保留原声），并对每条音轨做响度归一化
    # 注：如果有 N 个重击点，会产生 N+1 个时间片段
    # 按需加载足够的视频来覆盖所有片段（不循环）
    video_clips = []
    num_segments = len(hit_times) + 1  # 片段总数
    num_clips_needed = min(len(video_paths), num_segments)  # 取较小值
    videos_to_load = video_paths[:num_clips_needed]
    
    print(f"[加载视频] 检测到 {len(hit_times)} 个重击点 → {num_segments} 个时间片段")
    print(f"[加载视频] 共需 {num_clips_needed} 个视频素材，实际加载 {num_clips_needed} 个")
    
    for p in videos_to_load:
        clip = VideoFileClip(p)
        if clip.audio is not None:
            clip = clip.fx(audio_normalize)
        video_clips.append(clip)

    # 4. 按时间片段轮流从视频中取 subclip，每个片段对应一个视频（线性分配）
    result_clips = []
    
    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        seg_duration = end - start
        if seg_duration <= 0:
            continue

        # 每个片段对应一个视频：片段 N 用视频 N（如果超出范围则用最后一个视频）
        current_video_idx = min(seg_idx, len(video_clips) - 1)
        current_offset_in_video = 0.0
        
        print(f"[片段 {seg_idx}] 时间 {start:.2f}s-{end:.2f}s，使用视频 {current_video_idx} ({video_paths[current_video_idx].split(chr(92))[-1]})")

        # 从当前视频截取足够长的片段
        v = video_clips[current_video_idx]
        remain = v.duration - current_offset_in_video

        if remain >= seg_duration - 1e-3:
            # 当前素材够长，直接截取
            subclip = v.subclip(
                current_offset_in_video,
                current_offset_in_video + seg_duration,
            )
            result_clips.append(subclip)
        else:
            # 当前素材不够长，只从这个视频取，不去下一个视频
            # （因为下一个片段会用下一个视频）
            if remain > 1e-3:
                subclip = v.subclip(0, v.duration)
            else:
                # 如果视频太短，用整个视频
                subclip = v
            result_clips.append(subclip)

    # 5. 拼接所有视频片段（此时每段都带自己的原声）
    final_video = concatenate_videoclips(result_clips, method="compose")

    # 6. 处理音频：保留原声 + BGM 混音
    video_audio = final_video.audio  # 所有素材拼接后的原声
    if video_audio is not None:
        video_audio = video_audio.volumex(video_volume)

    # BGM 裁剪到成片时长，并调节音量
    bgm_for_mix = bgm_full.subclip(0, final_video.duration).volumex(bgm_volume)

    if video_audio is not None:
        mixed_audio = CompositeAudioClip([video_audio, bgm_for_mix])
    else:
        # 极端情况：素材里完全没有音频轨，就只用 BGM
        mixed_audio = bgm_for_mix

    final_video = final_video.set_audio(mixed_audio)

    # 7. 导出视频
    final_video.write_videofile(
        output_path,
        fps=fps,
        audio_codec="aac",
        codec="libx264",
    )

    # 8. 释放资源
    for v in video_clips:
        v.close()
    bgm_full.close()
    final_video.close()
