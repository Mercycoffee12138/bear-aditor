# audio_analysis.py
"""
音频分析模块：负责从 BGM 中检测“重击点”（beat-like hits）。

思路：
1. 读取音频为单声道，保留原采样率。
2. 计算 onset_strength（瞬态强度）和 RMS（响度）。
3. 将两者归一化并加权融合，得到一个综合“重击分数”序列。
4. 使用平滑 + 鲁棒阈值（median + k * MAD）筛选出强峰。
5. 应用最小时间间隔约束，只保留少量“大的重击”。

返回值：
- 一个以秒为单位的重击时间数组（numpy.ndarray）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

import pyloudnorm as pyln


@dataclass
class AudioAnalysisConfig:
    """音频分析配置（这里参数不暴露到命令行，方便以后再改）"""
    frame_length_ms: float = 46.0
    hop_length_ms: float = 23.0

    # 综合分数 = w_onset * onset_norm + w_rms * rms_norm
    onset_weight: float = 0.65
    rms_weight: float = 0.35

    # 阈值 = median + strict_factor * MAD（MAD 是中位数绝对偏差）
    strict_factor: float = 3.0  # 越大越“严格”，重击越少

    # 最小重击间隔（秒）—— 控制重击数量
    min_hit_interval: float = 0.6

    # 高斯平滑
    smooth_sigma: float = 1.0


def detect_strong_hits(
    audio_path: str,
    config: AudioAnalysisConfig | None = None,
) -> np.ndarray:
    """
    检测音频中的重击时间点（秒）。

    :param audio_path: 音频文件路径
    :param config: 可选配置对象
    :return: 以秒为单位的 np.ndarray，例如 array([0.5, 1.2, 2.8, ...])
    """
    if config is None:
        config = AudioAnalysisConfig()

    # 1. 读取音频：保留原采样率，强制单声道（更稳定）

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    # LUFS响度归一化，统一所有音频的响度标准
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    target_lufs = -14.0  # 可调整目标响度
    y = pyln.normalize.loudness(y, loudness, target_lufs)

    frame_length = int(config.frame_length_ms / 1000.0 * sr)
    hop_length = int(config.hop_length_ms / 1000.0 * sr)

    # 2. 计算 onset_strength（瞬态强度）
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    # 3. 计算 RMS 能量
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # 时间轴
    frames = np.arange(len(onset_env))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # 4. 归一化
    def safe_norm(x: np.ndarray) -> np.ndarray:
        if np.allclose(x.max(), 0):
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    onset_norm = safe_norm(onset_env)
    rms_norm = safe_norm(rms)

    # 5. 综合分数：偏向 onset，多一点“打击感”
    score = config.onset_weight * onset_norm + config.rms_weight * rms_norm

    # 6. 平滑
    score_smooth = gaussian_filter1d(score, sigma=config.smooth_sigma)

    # 7. 计算鲁棒阈值：median + k * MAD
    median = np.median(score_smooth)
    mad = np.median(np.abs(score_smooth - median)) + 1e-8
    threshold = median + config.strict_factor * mad

    candidate_idx = np.where(score_smooth > threshold)[0]

    if candidate_idx.size == 0:
        # 极端情况：没有任何点超过阈值，兜底为返回空数组
        return np.array([], dtype=float)

    # 8. 应用最小间隔：在给定窗口内只保留分数最大的一个
    selected_indices: List[int] = []
    last_time = -1e9

    for idx in candidate_idx:
        t = times[idx]
        if selected_indices and (t - last_time) < config.min_hit_interval:
            # 如果和上一个选中的点太近，则保留分数更大的那个
            if score_smooth[idx] > score_smooth[selected_indices[-1]]:
                selected_indices[-1] = idx
                last_time = t  # 同时更新时间
            continue

        selected_indices.append(idx)
        last_time = t

    hit_times = times[selected_indices]

    return hit_times
