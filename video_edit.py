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

from dataclasses import dataclass
from typing import List, Sequence
from pathlib import Path

import numpy as np
import librosa
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeAudioClip,
    vfx,
)
from moviepy.audio.fx.all import audio_normalize


def apply_transition(clip, transition_type: str = "fade", duration: float = 0.3):
    """
    对视频片段应用转场特效。
    
    :param clip: 视频片段
    :param transition_type: 转场类型 ("fade"淡入淡出, "zoom"缩放淡出)
    :param duration: 转场时长（秒）
    :return: 应用了转场特效的视频片段
    """
    if transition_type == "fade":
        # 淡入淡出效果
        return clip.fx(vfx.fadeout, duration)
    elif transition_type == "zoom":
        # 缩放淡出效果
        return clip.fx(vfx.resize, lambda t: 1 + 0.02 * t).fx(vfx.fadeout, duration)
    else:
        return clip


@dataclass
class SpeechSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_speech_segments(
    video_path: str,
    vad_aggressiveness: int = 2,
    frame_ms: int = 30,
    min_duration: float = 0.25,
    min_silence: float = 0.2,
    sr_target: int = 16000,
) -> List[SpeechSegment]:
    """
    使用 webrtcvad 做语音段检测（只做人声/说话，不识别文字）。
    - 只保留不短于 min_duration 的语音段。
    - 相隔小于 min_silence 的段会合并。
    """
    try:
        import webrtcvad
    except ImportError as e:
        raise ImportError("需要安装 webrtcvad，可先执行: pip install webrtcvad") from e

    y, sr = librosa.load(video_path, sr=sr_target, mono=True)
    if y.size == 0:
        return []

    # webrtcvad 需要 16-bit PCM bytes
    pcm = np.clip(y, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)

    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_length = int(sr * frame_ms / 1000)
    if frame_length <= 0:
        frame_length = int(sr * 0.03)

    segments: List[SpeechSegment] = []
    in_seg = False
    seg_start = 0.0

    for idx in range(0, len(pcm16) - frame_length + 1, frame_length):
        frame = pcm16[idx : idx + frame_length]
        is_speech = vad.is_speech(frame.tobytes(), sample_rate=sr)
        t = idx / sr
        if is_speech and not in_seg:
            in_seg = True
            seg_start = t
        elif not is_speech and in_seg:
            seg_end = t
            if seg_end - seg_start >= min_duration:
                segments.append(SpeechSegment(seg_start, seg_end))
            in_seg = False

    # 收尾：文件结尾仍在语音段中
    if in_seg:
        seg_end = len(pcm16) / sr
        if seg_end - seg_start >= min_duration:
            segments.append(SpeechSegment(seg_start, seg_end))

    # 合并间隔很短的段
    merged: List[SpeechSegment] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start - prev.end <= min_silence:
            merged[-1] = SpeechSegment(prev.start, seg.end)
        else:
            merged.append(seg)

    return merged


def select_segment_for_slice(
    speech_segments: Sequence[SpeechSegment],
    slice_duration: float,
) -> SpeechSegment | None:
    """选择“时长不大于 slice 且最长”的语音段，找不到则返回 None。"""
    if not speech_segments:
        return None
    # 先过滤不大于 slice_duration 的
    candidates = [s for s in speech_segments if s.duration <= slice_duration + 1e-6]
    if candidates:
        return max(candidates, key=lambda s: s.duration)
    # 没有更短的，退化为整个列表中最长的
    return max(speech_segments, key=lambda s: s.duration)


def select_best_segment_across_videos(
    speech_segments_map: Sequence[Sequence[SpeechSegment]],
    slice_duration: float,
    used_segments: set[tuple[int, int]],
    used_videos: set[int],
    last_video_idx: int = -1,
) -> tuple[int, int, SpeechSegment] | None:
    """
    在所有视频的语音段中，为当前时间片挑选最佳语音段。
    仅考虑"时长不超过 slice_duration"的段，越接近 slice 越好。
    避免选择与上一个相邻片段使用的同一视频。
    返回 (video_idx, seg_idx, segment)；找不到则返回 None。
    """
    candidates: list[tuple[tuple[int, float, float, float, float, float], tuple[int, int, SpeechSegment]]] = []

    for vid_idx, segs in enumerate(speech_segments_map):
        for seg_idx, seg in enumerate(segs):
            if (vid_idx, seg_idx) in used_segments:
                continue  # 不允许重复片段
            diff = slice_duration - seg.duration
            if diff < -1e-6:
                continue  # 只要不长于音频时间片
            over_flag = 0
            key_len = -seg.duration  # 越接近 slice（越长）越好
            video_used_flag = 1 if vid_idx in used_videos else 0
            # 如果是上一个视频，优先级降低（加大权重）
            adjacent_flag = 1 if vid_idx == last_video_idx else 0
            key = (adjacent_flag, video_used_flag, over_flag, key_len, -seg.duration, float(vid_idx))
            candidates.append((key, (vid_idx, seg_idx, seg)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def build_beat_synced_video(
    audio_path: str,
    video_paths: List[str],
    hit_times: np.ndarray,
    output_path: str,
    fps: int = 25,
    bgm_volume: float = 0.5,
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
    speech_segments_map: List[List[SpeechSegment]] = []
    num_segments = len(hit_times) + 1  # 片段总数

    print(f"[加载视频] 检测到 {len(hit_times)} 个重击点 → {num_segments} 个时间片段")
    print(f"[加载视频] 共需 {num_segments} 段素材，实际加载 {len(video_paths)} 个视频（全量）")
    
    for p in video_paths:
        clip = VideoFileClip(p)
        if clip.audio is not None:
            clip = clip.fx(audio_normalize)
        video_clips.append(clip)
        segs = detect_speech_segments(p)
        speech_segments_map.append(segs)
        print(f"[语音检测] {p} 检出 {len(segs)} 段语音")

    # 4. 按时间片段轮流从视频中取 subclip，每个片段对应一个视频（线性分配）
    result_clips = []
    used_segments: set[tuple[int, int]] = set()  # (video_idx, seg_idx)
    used_videos: set[int] = set()
    last_video_idx = -1  # 跟踪上一个视频的索引
    last_clip_start = 0.0  # 跟踪上一个片段的起始时间
    last_v = None  # 跟踪上一个视频对象
    last_used_previous_video = False  # 跟踪上一个片段是否使用了前一个视频的延伸
    min_segment_duration = 1.0  # 最小片段时长阈值
    
    # 预先计算第一个片段的时长
    first_segment_start = segment_starts[0]
    first_segment_end = segment_ends[0]
    first_segment_duration = first_segment_end - first_segment_start
    
    # 如果第一个片段过短，则与第二个片段合并
    if first_segment_duration < min_segment_duration and len(segment_starts) > 1:
        # 合并第一个和第二个片段
        segment_starts = [segment_starts[0]] + segment_starts[2:]
        segment_ends = [segment_ends[1]] + segment_ends[2:]
        print(f"[预处理] 第一个片段过短（{first_segment_duration:.2f}s < {min_segment_duration}s），与第二个片段合并")
    
    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        seg_duration = end - start
        if seg_duration <= 0:
            continue

        # 检查是否需要用上一个片段补充（但第一个片段不采用此策略，保证第一个是长片段）
        # 同时避免连续两个片段都使用上一个视频的延伸
        use_previous = False
        if (seg_idx > 0 and seg_duration < min_segment_duration and result_clips and last_v is not None 
            and not last_used_previous_video):  # 上一个片段没有使用前一个视频
            # 优先尝试从上一个片段向后延伸
            extension_available = last_v.duration - last_clip_start
            if extension_available > seg_duration:
                use_previous = True

        if use_previous:
            # 用上一个片段向后延伸
            clip_start = last_clip_start + (result_clips[-1].duration if result_clips[-1].duration else 0)
            clip_end = min(last_v.duration, clip_start + seg_duration)
            actual_length = clip_end - clip_start
            
            # 如果向后延伸不够，则向前调整（向前延伸）
            if actual_length < seg_duration:
                shift_back = seg_duration - actual_length
                clip_start = max(0.0, clip_start - shift_back)
                clip_end = min(last_v.duration, clip_start + seg_duration)
            
            current_video_idx = last_video_idx
            v = last_v
            selected = None
            video_name = Path(video_paths[current_video_idx]).name
            last_used_previous_video = True  # 标记此片段使用了上一个视频的延伸
            
            print(
                f"[片段 {seg_idx}] 时间 {start:.2f}s-{end:.2f}s（短片段，用上一个），"
                f"视频索引 {current_video_idx}（{video_name}）→ 截取 {clip_start:.2f}-{clip_end:.2f}s"
            )
        else:
            # 在所有视频的语音段中寻找最优匹配
            best = select_best_segment_across_videos(
                speech_segments_map, seg_duration, used_segments, used_videos, last_video_idx
            )
            if best is not None:
                vid_idx, speech_idx, selected = best
                used_segments.add((vid_idx, speech_idx))
                used_videos.add(vid_idx)

            if best is None:
                # 找不到任何语音段，退化为线性分配
                # 尽量使用尚未使用过的视频，且不使用上一个相邻片段的视频
                unused_videos = [
                    idx for idx in range(len(video_clips)) 
                    if idx not in used_videos and idx != last_video_idx
                ]
                if unused_videos:
                    current_video_idx = unused_videos[0]
                else:
                    # 如果都用过了，找一个不是上一个的视频
                    other_videos = [idx for idx in range(len(video_clips)) if idx != last_video_idx]
                    current_video_idx = other_videos[seg_idx % len(other_videos)] if other_videos else seg_idx % len(video_clips)
                v = video_clips[current_video_idx]
                selected = None
                used_videos.add(current_video_idx)
            else:
                current_video_idx, _, selected = best
                v = video_clips[current_video_idx]

            video_name = Path(video_paths[current_video_idx]).name

            if selected is None:
                # 没有语音段，退化为从头截取
                clip_start = 0.0
                clip_end = min(v.duration, seg_duration)
                padding = seg_duration - (clip_end - clip_start)
                if padding > 1e-3 and clip_end < v.duration:
                    clip_end = min(v.duration, clip_end + padding)
            else:
                # 左对齐：从语音起点开始，优先向后扩展
                clip_start = max(0.0, selected.start)
                clip_end = clip_start + seg_duration
                if clip_end > v.duration:
                    # 不够长则向前微调以满足时长
                    shift_back = clip_end - v.duration
                    clip_start = max(0.0, clip_start - shift_back)
                    clip_end = min(v.duration, clip_start + seg_duration)

            print(
                f"[片段 {seg_idx}] 时间 {start:.2f}s-{end:.2f}s，"
                f"用视频索引 {current_video_idx}（{video_name}），选语音段 "
                f"{'无' if selected is None else f'{selected.start:.2f}-{selected.end:.2f}s'} → "
                f"截取 {clip_start:.2f}-{clip_end:.2f}s"
            )
            
            last_video_idx = current_video_idx
            last_clip_start = clip_start
            last_v = v
            last_used_previous_video = False  # 重置标志，因为使用了新视频

        subclip = v.subclip(clip_start, clip_end)
        
        # 如果有向后扩展，对扩展部分进行静音（但扩展比例不能太大）
        if selected is not None and selected.end > 0:
            expansion_start = selected.end - clip_start  # 相对于 subclip 的时间位置
            if expansion_start < subclip.duration and expansion_start > 0:
                # 计算扩展部分的占比
                expansion_ratio = (subclip.duration - expansion_start) / subclip.duration
                # 只有当扩展部分不超过 30% 时才进行静音
                if expansion_ratio <= 0.3 and subclip.audio is not None:
                    audio_start = subclip.audio.subclip(0, expansion_start)
                    audio_end_silent = subclip.audio.subclip(expansion_start, subclip.duration).volumex(0)
                    
                    # 使用 CompositeAudioClip 拼接音频
                    final_audio = CompositeAudioClip([audio_start, audio_end_silent])
                    subclip = subclip.set_audio(final_audio)
        
        result_clips.append(subclip)

    # 5. 为每个片段（除最后一个）添加淡出转场特效
    # 但如果片段时长太短，就不使用特效（避免频繁切换时特效突兀）
    transition_duration = 0.2  # 转场时长（秒）
    min_duration_for_transition = 2  # 片段最小时长阈值
    
    for i in range(len(result_clips) - 1):
        # 只为足够长的片段添加转场
        if result_clips[i].duration >= min_duration_for_transition:
            result_clips[i] = result_clips[i].fx(vfx.fadeout, transition_duration)
    
    # 拼接所有视频片段（此时每段都带自己的原声）
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
