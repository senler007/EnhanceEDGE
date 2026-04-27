import os
import argparse
import numpy as np
import librosa
from scipy.signal import find_peaks
import pickle

FPS = 30

def load_motion_energy(motion_path):
    """
    motion:
    - .npy: [T, J, C] 或 [T, D]
    - .pkl: EDGE 保存结果，读取 info["full_pose"]，通常是 [T, J, 3]
    """
    if motion_path.endswith(".npy"):
        motion = np.load(motion_path)
    elif motion_path.endswith(".pkl"):
        with open(motion_path, "rb") as f:
            info = pickle.load(f)
        motion = info["full_pose"]
    else:
        raise ValueError(f"Unsupported motion file type: {motion_path}")

    if motion.ndim == 3:
        vel = np.diff(motion, axis=0)
        energy = np.linalg.norm(vel, axis=-1).mean(axis=-1)
    elif motion.ndim == 2:
        vel = np.diff(motion, axis=0)
        energy = np.linalg.norm(vel, axis=-1)
    else:
        raise ValueError(f"Unsupported motion shape: {motion.shape}")

    return energy


def extract_motion_beats(energy, distance=10, prominence=None):
    """
    从动作能量中提取峰值点，作为 motion beats
    distance: 峰值最小间隔（帧）
    prominence: 峰值显著性，可为空自动估计
    """
    if prominence is None:
        prominence = max(1e-6, np.std(energy) * 0.5)

    peaks, _ = find_peaks(energy, distance=distance, prominence=prominence)
    return peaks


def extract_music_beats(audio_path):
    """
    用 librosa 提取音乐 beat 时间点（秒）
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, tempo


def beat_alignment_score(motion_peaks, music_beats, fps=30, sigma=0.1):
    """
    motion_peaks: 动作峰值帧号
    music_beats: 音乐 beat 时间（秒）
    sigma: 时间容忍度，越小越严格

    公式思路：
    对每个 motion beat，找最近的 music beat，
    根据时间差 dt 计算 exp(-dt^2 / (2*sigma^2))
    然后取平均，越接近 1 越好
    """
    if len(motion_peaks) == 0 or len(music_beats) == 0:
        return 0.0

    motion_times = motion_peaks / fps
    scores = []

    for mt in motion_times:
        dt = np.min(np.abs(music_beats - mt))
        score = np.exp(-(dt ** 2) / (2 * sigma ** 2))
        scores.append(score)

    return float(np.mean(scores))


def evaluate_pair(motion_path, audio_path):
    energy = load_motion_energy(motion_path)
    motion_peaks = extract_motion_beats(energy)
    music_beats, tempo = extract_music_beats(audio_path)
    ba = beat_alignment_score(motion_peaks, music_beats, fps=FPS)

    return {
        "motion_file": os.path.basename(motion_path),
        "audio_file": os.path.basename(audio_path),
        "tempo": float(tempo),
        "num_motion_beats": int(len(motion_peaks)),
        "num_music_beats": int(len(music_beats)),
        "ba": ba,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="eval/motions/",
        help="生成动作npy目录,默认 eval/motions/"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="custom_music",
        help="对应wav目录,默认 data/test/wavs/"
    )
    parser.add_argument(
        "--save_txt",
        type=str,
        default="eval/beat_align_result.txt",
        help="保存结果txt,默认 eval/beat_align_result.txt"
    )
    args = parser.parse_args()

    motion_files = sorted([
        f for f in os.listdir(args.motion_dir)
        if f.endswith(".npy") or f.endswith(".pkl")
    ])

    if len(motion_files) == 0:
        raise ValueError(f"No npy files found in {args.motion_dir}")

    results = []
    for mf in motion_files:
        stem = os.path.splitext(mf)[0]

        # 兼容 test_000005.pkl -> 000005.wav
        if stem.startswith("test_"):
            audio_stem = stem.replace("test_", "", 1)
        else:
            audio_stem = stem

        motion_path = os.path.join(args.motion_dir, mf)
        audio_path = os.path.join(args.audio_dir, audio_stem + ".wav")

        if not os.path.exists(audio_path):
            print(f"[Skip] audio not found for {mf}: {audio_path}")
            continue

        try:
            res = evaluate_pair(motion_path, audio_path)
            results.append(res)
            print(f"{mf} | BA={res['ba']:.4f} | tempo={res['tempo']:.2f}")
        except Exception as e:
            print(f"[Error] {mf}: {e}")

    if len(results) == 0:
        print("No valid motion-audio pairs found.")
        return

    mean_ba = np.mean([r["ba"] for r in results])
    print("=" * 50)
    print(f"Average Beat Alignment: {mean_ba:.4f}")
    print("=" * 50)

    if args.save_txt is not None:
        with open(args.save_txt, "w", encoding="utf-8") as f:
            for r in results:
                f.write(
                    f"{r['motion_file']}\t{r['audio_file']}\t"
                    f"tempo={r['tempo']:.2f}\t"
                    f"motion_beats={r['num_motion_beats']}\t"
                    f"music_beats={r['num_music_beats']}\t"
                    f"BA={r['ba']:.4f}\n"
                )
            f.write("=" * 50 + "\n")
            f.write(f"Average Beat Alignment: {mean_ba:.4f}\n")


if __name__ == "__main__":
    main()