import os
from functools import partial
from pathlib import Path

import jukemirlib
import numpy as np
from tqdm import tqdm

FPS = 30
LAYER = 66 # Jukebox不同层提取的特征语义层次不同。低层主要包含音频信号级特征，高层更偏向音乐生成语义，而中高层能够同时捕捉节奏、节拍以及音乐结构信息。已有研究表明第66层的特征对舞蹈生成任务表现较好，因此本文采用Jukebox第66层作为音乐特征表示。


def extract(fpath, skip_completed=True, dest_dir="aist_juke_feats"):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    audio = jukemirlib.load_audio(fpath) # 将wav文件转换为声音原始振幅序列
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS) # 把声音变成 神经网络里的高维音乐表征

    #np.save(save_path, reps[LAYER])
    return reps[LAYER], save_path # 得到的是音乐的特征向量 (第几帧,特征向量)  


def extract_folder(src, dest):
    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, skip_completed=False, dest_dir=dest)
    for fpath in tqdm(fpaths):
        rep, path = extract_(fpath)
        np.save(path, rep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")

    args = parser.parse_args()

    extract_folder(args.src, args.dest)
