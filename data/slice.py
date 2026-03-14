import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


import os
import math
import numpy as np
import librosa as lr
import soundfile as sf


def slice_audio(audio_file, stride, length, out_dir):
    """
    按滑窗切片，最后一段允许补零。
    以“覆盖到音频末尾”为停止条件。
    """
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    window = int(length * sr)
    stride_step = int(stride * sr)
    audio_len = len(audio)

    # 需要多少个窗口，才能覆盖到音频结尾
    if audio_len <= window:
        num_slices = 1
    else:
        num_slices = math.ceil((audio_len - window) / stride_step) + 1

    for idx in range(num_slices):
        start_idx = idx * stride_step
        end_idx = start_idx + window

        audio_slice = np.zeros(window, dtype=audio.dtype)

        if start_idx < audio_len:
            valid_end = min(end_idx, audio_len)
            valid_len = valid_end - start_idx
            audio_slice[:valid_len] = audio[start_idx:valid_end]

        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)

    return num_slices


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
