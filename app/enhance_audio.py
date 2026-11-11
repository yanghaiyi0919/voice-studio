# enhance_audio.py
# 语音增强模块（谱减 + VAD/静音裁剪 + 低能放大 + 断续修补 + 可视化）
# 输出（默认生成）:
#   output_enhanced.wav
#   preview_wave.png               (波形对比)
#   preview_spectrogram.png        (频谱对比)
#   noise_analysis.png             (噪声功率谱估计)

import os
import numpy as np
from scipy import signal
from scipy.signal import stft, istft, medfilt
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

# ---------- 默认参数 ----------
SR = 16000
N_FFT = 1024
HOP = 256
WIN = 'hann'
NOISE_FRAMES_SEC = 0.5    # 用前 0.5s 估计噪声
SMOOTH_MEDIAN_K = 3
RMS_LOW_THRESH = 0.02
LOW_GAIN_BASE = 1.8
GAP_MS_THRESHOLD = 30
MIN_ENERGY = 1e-9

# ---------- 工具函数 ----------
def load_wav(path_or_array, sr=SR):
    """
    如果传入是 numpy array (audio, sr)，则返回 array。
    如果传入是路径则读取文件并返回 (array, sr).
    """
    if isinstance(path_or_array, str):
        data, fs = sf.read(path_or_array)
        if fs != sr:
            # resample
            num = int(len(data) * sr / fs)
            data = signal.resample(data, num)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32)
    else:
        # assume tuple (sr, np.array) or numpy array
        if isinstance(path_or_array, tuple) or isinstance(path_or_array, list):
            sr_in, arr = path_or_array
            if sr_in != sr:
                num = int(len(arr) * sr / sr_in)
                arr = signal.resample(arr, num)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            return arr.astype(np.float32)
        else:
            arr = np.asarray(path_or_array, dtype=np.float32)
            return arr

def save_wav(path, data, sr=SR):
    # limit amplitude
    mx = np.max(np.abs(data)) + 1e-12
    if mx > 1.0:
        data = data / mx * 0.99
    sf.write(path, data.astype(np.float32), sr, subtype='PCM_16')

def trim_silence(x, sr=SR, top_db=40, frame_length=1024, hop_length=256):
    """简单的短时能量静音裁剪（去前后静音）"""
    # compute frame energy
    energy = []
    for i in range(0, max(1, len(x)-frame_length+1), hop_length):
        frame = x[i:i+frame_length]
        energy.append(np.sum(frame**2))
    energy = np.array(energy)
    if energy.size == 0:
        return x
    eps = 1e-12
    # threshold relative to median
    med = np.median(energy) + eps
    thresh = med / (10**(top_db/10.0))  # top_db -> ratio
    # find first frame above threshold
    idxs = np.where(energy > thresh)[0]
    if idxs.size == 0:
        return x  # all silence
    start_frame = max(0, idxs[0]-1)
    end_frame = min(len(energy)-1, idxs[-1]+1)
    start_sample = start_frame * hop_length
    end_sample = min(len(x), (end_frame+1)*hop_length)
    return x[start_sample:end_sample]

def simple_vad(x, sr=SR, frame_ms=30, energy_threshold_ratio=0.3):
    """
    简单能量 VAD，返回 mask (same length as x) 表示候选声音活动。
    frame_ms: 帧长（ms）
    energy_threshold_ratio: 相对于全局中位能量的阈值
    """
    frame_len = int(frame_ms/1000.0 * sr)
    hop = frame_len // 2
    frames = []
    for i in range(0, max(1, len(x)-frame_len+1), hop):
        frame = x[i:i+frame_len]
        frames.append(frame)
    if len(frames) == 0:
        return np.ones_like(x, dtype=bool)
    energies = np.array([np.sqrt(np.mean(f**2)+1e-12) for f in frames])
    med = np.median(energies)+1e-12
    thresh = med * energy_threshold_ratio
    vad_frames = energies > thresh
    # expand to sample-level mask
    mask = np.zeros(len(x), dtype=bool)
    idx = 0
    for i in range(0, max(1, len(x)-frame_len+1), hop):
        if vad_frames[idx]:
            mask[i:i+frame_len] = True
        idx += 1
    # if nothing detected, fallback to all True
    if mask.sum() == 0:
        mask[:] = True
    return mask

def estimate_noise_mag(mag, sr=SR, n_frames_sec=NOISE_FRAMES_SEC):
    frames = int(np.ceil((n_frames_sec * sr - (N_FFT - HOP)) / HOP))
    frames = max(1, frames)
    noise_mag = np.mean(mag[:, :frames], axis=1, keepdims=True)
    return noise_mag

def spectral_subtraction(mag, noise_mag, alpha=1.0, flooring=1e-8):
    # mag: (freq_bins, frames)
    mag_sub = mag - alpha * noise_mag
    mag_sub = np.maximum(mag_sub, flooring)
    return mag_sub

def spectral_smoothing(mag, k=SMOOTH_MEDIAN_K):
    if k > 1:
        k = k if k % 2 else k + 1
        return medfilt(mag, kernel_size=(1, k))
    return mag

def fix_dropouts(x, sr=SR, gap_ms=GAP_MS_THRESHOLD):
    abs_x = np.abs(x)
    mask = abs_x < 1e-4
    gap_samples = int(gap_ms/1000.0 * sr)
    out = x.copy()
    i = 0
    N = len(x)
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            gap_len = j - i
            if gap_len >= gap_samples:
                left = i-1
                right = j
                if left < 0:
                    out[i:right] = 0.0
                elif right >= N:
                    out[left+1:N] = out[left]
                else:
                    out[left+1:right] = np.linspace(out[left], out[right], gap_len)
            i = j
        else:
            i += 1
    return out

def amplify_low_energy(x, sr=SR, strength=1.0):
    frame_len = N_FFT
    hop = HOP
    frames_rms = []
    for i in range(0, max(1, len(x)-frame_len+1), hop):
        f = x[i:i+frame_len]
        frames_rms.append(np.sqrt(np.mean(f**2)+1e-12))
    frames_rms = np.array(frames_rms)
    if frames_rms.size == 0:
        return x
    median_rms = np.median(frames_rms)+1e-12
    out = x.copy()
    idx = 0
    for i in range(0, max(1, len(x)-frame_len+1), hop):
        rms = frames_rms[idx]
        if rms < RMS_LOW_THRESH * median_rms:
            gain = 1.0 + (LOW_GAIN_BASE - 1.0) * strength
            out[i:i+frame_len] = out[i:i+frame_len] * (1.0 + (gain-1.0) * 0.5)
        idx += 1
    return out

def combine_trim(a, b):
    L = min(len(a), len(b))
    return a[:L], b[:L]

# ---------- 可视化 ----------
def plot_waveform_and_spectrogram(x_before, x_after, sr=SR, out_wave='preview_wave.png', out_spec='preview_spectrogram.png'):
    # Waveform figure
    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1)
    t1 = np.linspace(0, len(x_before)/sr, len(x_before))
    plt.plot(t1, x_before, linewidth=0.5)
    plt.title("Original waveform")
    plt.xlabel("Time (s)")
    plt.subplot(2,2,2)
    t2 = np.linspace(0, len(x_after)/sr, len(x_after))
    plt.plot(t2, x_after, linewidth=0.5)
    plt.title("Enhanced waveform")
    plt.xlabel("Time (s)")

        # --- Spectrogram Compare (Original vs Enhanced) ---
    plt.subplot(2,2,3)
    f, t, Z = stft(x_before, fs=sr, window=WIN, nperseg=N_FFT, noverlap=N_FFT-HOP, boundary=None)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Z)+1e-9), shading='gouraud')
    plt.title("Original Spectrogram (dB)")
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Time (s)")

    plt.subplot(2,2,4)
    f, t, Z = stft(x_after, fs=sr, window=WIN, nperseg=N_FFT, noverlap=N_FFT-HOP, boundary=None)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Z)+1e-9), shading='gouraud')
    plt.title("Enhanced Spectrogram (dB)")
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_wave, dpi=150)
    plt.close()

    # Save spectrogram as separate file (higher contrast)
    plt.figure(figsize=(10,4))
    f, t, Z = stft(x_before, fs=sr, window=WIN, nperseg=N_FFT, noverlap=N_FFT-HOP, boundary=None)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Z)+1e-9), shading='gouraud')
    plt.title("Original Spectrogram (dB)")
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_spec.replace('.png','_orig.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10,4))
    f, t, Z = stft(x_after, fs=sr, window=WIN, nperseg=N_FFT, noverlap=N_FFT-HOP, boundary=None)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Z)+1e-9), shading='gouraud')
    plt.title("Enhanced Spectrogram (dB)")
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_spec.replace('.png','_enh.png'), dpi=150)
    plt.close()

def plot_noise_analysis(noise_mag, sr=SR, out_png='noise_analysis.png'):
    plt.figure(figsize=(8,4))
    freq = np.linspace(0, sr/2, noise_mag.shape[0])
    plt.semilogy(freq, np.squeeze(noise_mag), linewidth=1.0)
    plt.title("Estimated noise magnitude (linear scale)")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ---------- 主入口 ----------
def enhance_array(audio_input, sr=SR, strength=0.8, use_vad=True, do_trim=True, output_dir=None):
    """
    输入可以是 numpy array 或者 (sr, array)
    strength: 0.0 - 1.0 (增强强度)
    use_vad: 是否使用 VAD 过滤非语音段进行噪声估计
    do_trim: 是否去前后静音（裁剪）
    返回: enhanced_array, dict_of_paths
    """
    x = load_wav(audio_input, sr=sr)

    # 1) 可选静音裁剪（提高噪声估计准确性）
    if do_trim:
        x_trim = trim_silence(x, sr=sr)
    else:
        x_trim = x

    # 2) VAD（用于只用语音段估计噪声/或仅对语音段增强）
    vad_mask = None
    if use_vad:
        vad_mask = simple_vad(x_trim, sr=sr)
    else:
        vad_mask = np.ones_like(x_trim, dtype=bool)

    # 3) STFT
    f, t, Zxx = stft(x_trim, fs=sr, window=WIN, nperseg=N_FFT, noverlap=N_FFT-HOP, boundary="zeros")
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # 4) 噪声估计（仅用 vad_mask==False 的帧或前若干帧）
    # 简单：先用前 NOISE_FRAMES_SEC 秒；若VAD可用则用 VAD 标注为非语音的帧
    noise_mag = estimate_noise_mag(mag, sr=sr, n_frames_sec=NOISE_FRAMES_SEC)
    # 如果 VAD 标注出静音帧，我们可以用那些帧更准确估计
    if use_vad:
        # 收集被标注为非语音的 frames
        # 构建 frame-level vad (len(t) frames)
        frame_vad = []
        frame_len = int(N_FFT)
        hop = HOP
        idx = 0
        for i in range(0, max(1, len(x_trim)-frame_len+1), hop):
            frame_vad.append(vad_mask[i:i+frame_len].any())
            idx += 1
        frame_vad = np.array(frame_vad[:mag.shape[1]])
        silent_frames_idx = np.where(frame_vad == False)[0]
        if silent_frames_idx.size > 0:
            # use these frames to estimate noise
            noise_mag2 = np.mean(mag[:, silent_frames_idx], axis=1, keepdims=True)
            # combine estimates (mean)
            noise_mag = 0.5 * noise_mag + 0.5 * noise_mag2

    # 5) 谱减（alpha 随 strength 调节）
    alpha = 0.8 + 1.2 * strength   # alpha in [0.8, 2.0]
    mag_sub = spectral_subtraction(mag, noise_mag, alpha=alpha, flooring=1e-8)
    mag_sub = spectral_smoothing(mag_sub, k=max(1, int(SMOOTH_MEDIAN_K * strength)))

    # 6) 用相位重构回时域
    Z_enh = mag_sub * np.exp(1j * phase)
    _, x_enh = istft(Z_enh, fs=sr, window=WIN, nperseg=N_FFT, noverlap=N_FFT-HOP, input_onesided=True, boundary=True)

    # 7) 修补断续
    x_fixed = fix_dropouts(x_enh, sr=sr)

    # 8) 对微弱声音放大（strength 控制强度）
    x_ampl = amplify_low_energy(x_fixed, sr=sr, strength=strength)

    # 9) 对增强结果做温和压缩/归一化，避免 clipping
    peak = np.max(np.abs(x_ampl)) + 1e-12
    if peak > 0:
        x_final = x_ampl / peak * 0.99
    else:
        x_final = x_ampl

    # 10) 将处理后的信号放回到与原始一致的长度（如果裁剪过，返回裁剪后长度）
    x_orig_cut, x_final_cut = combine_trim(x_trim, x_final)

    # 11) 保存文件并生成可视化
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(output_dir or ".")
    directory.mkdir(parents=True, exist_ok=True)
    out_wav = str(directory / f"output_enhanced_{time_tag}.wav")
    out_wave = str(directory / f"preview_wave_{time_tag}.png")
    out_spec = str(directory / f"preview_spectrogram_{time_tag}.png")
    out_noise = str(directory / f"noise_analysis_{time_tag}.png")

    save_wav(out_wav, x_final_cut, sr=sr)
    # wave + spectrograms
    plot_waveform_and_spectrogram(x_orig_cut, x_final_cut, sr=sr, out_wave=out_wave, out_spec=out_spec)
    # noise plot
    plot_noise_analysis(noise_mag, sr=sr, out_png=out_noise)

    info = {
        "wav": out_wav,
        "wave_png": out_wave,
        "spec_png": out_spec.replace('.png','_enh.png'),  # we saved _orig and _enh as separate files in the function
        "noise_png": out_noise,
        "noise_mag": noise_mag
    }
    return x_final_cut, info


# 简单封装：给 LASTTT.py 导入用
def enhance_file_from_path(inpath="output.wav", strength=0.8, use_vad=True, do_trim=True, output_dir=None):
    """
    直接从路径读文件并写出结果，返回路径和预览图路径
    """
    if not os.path.exists(inpath):
        raise FileNotFoundError("input file not found: "+str(inpath))
    x = load_wav(inpath, SR)
    x_enh, info = enhance_array(x, sr=SR, strength=strength, use_vad=use_vad, do_trim=do_trim, output_dir=output_dir)
    # info contains file names
    return info["wav"], info["wave_png"], info["spec_png"], info["noise_png"]


