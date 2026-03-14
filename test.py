# 调用模型生成结果 : 读取音乐 → 提取音乐特征 → 调用 EDGE 模型 → 生成舞蹈 → 保存结果
import glob
import math
import os
import librosa
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1]) # x = song_slice10.wav 时得到 10


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)

# 输入想要的秒数 , 自动算出需要多少个 5 秒片段
def get_required_slice_count(target_length, window=5.0, stride=2.5):
    """
    根据目标舞蹈长度，计算需要多少个 slice。
    总长度公式：
        total_length = window + (N - 1) * stride
    """
    if target_length is None:
        return None

    if target_length <= 0:
        raise ValueError("target_length must be > 0")

    if target_length <= window:
        return 1

    return math.ceil((target_length - window) / stride) + 1


def get_generated_length(slice_count, window=5.0, stride=2.5):
    """
    根据 slice 数量反推最终覆盖时长。
    """
    if slice_count <= 0:
        return 0.0
    return window + (slice_count - 1) * stride



# test 做的 : 读取 wav file 里的wav,剪成片段,随机选取片段
def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract # 选择音乐特征

    temp_dir_list = []
    all_cond = [] # 音乐特征
    all_filenames = [] # 切出来的音频文件
    all_target_frames = []

    # target_length = 
    # required_slice_count = get_required_slice_count(target_length, window=5.0, stride=2.5)

    # if target_length is not None:
    #     print(f"Target dance length: {target_length:.2f}s")
    #     print(f"Required slice count: {required_slice_count}")
    #     print(f"Actual generated length will be about: {get_generated_length(required_slice_count):.2f}s")

    # 使用缓存特征
    if opt.use_cached_features:
        print("Using precomputed features")
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))

        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            feat_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)

            assert len(file_list) == len(feat_file_list), f"wav/npy count mismatch in {dir}"

            file_list = file_list[0:-1]
            feat_file_list = feat_file_list[0:-1]

            cond_list = [np.load(x) for x in feat_file_list]
            cond_list = torch.from_numpy(np.array(cond_list))

            all_filenames.append(file_list)
            all_cond.append(cond_list)

    # 现算特征
    else:
        print("Computing features for input music")

        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # 计算每首音乐的长度
            audio, sr = librosa.load(wav_file, sr=None)
            audio_length = len(audio) / sr
            target_frames = int(round(audio_length * 30))

            print(f"Music: {wav_file}")
            print(f"Length: {audio_length:.2f}s")
            print(f"Target frames: {target_frames}")

            all_target_frames.append(target_frames)

            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name

            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)


            cond_list = []
            print(f"Computing features for {wav_file}")
            for file in tqdm(file_list):
                reps, _ = feature_func(file) # 提取特征
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)

                cond_list.append(reps)

            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list)

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")

    for i in range(len(all_cond)):
        target_frames = all_target_frames[i]

        print(f"Generating dance with {target_frames} frames")

        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple,
            "test",
            opt.render_dir,
            render_count=-1,
            fk_out=fk_out,
            render=not opt.no_render,
            target_frames=target_frames
        )

    print("Done")
    torch.cuda.empty_cache()

    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)