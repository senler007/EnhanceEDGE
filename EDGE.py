# 整个系统的 核心类，负责diffusion + dance decoder
##这段代码实现了基于 Accelerate 的多 GPU 训练流程。首先由主进程创建实验目录并初始化 wandb，用于记录超参数和训练日志。随后各进程同步进入 epoch 循环。每个 batch 中，模型根据动作数据和音乐条件进行前向传播，得到总损失及多个子损失，包括速度损失、前向运动学损失和脚部损失。之后清空梯度，执行反向###传播，并通过优化器更新参数。训练过程中主进程负责累计损失，并按设定间隔更新 EMA 模型，以获得更稳定的生成效果。每隔若干 epoch，主进程会保存 checkpoint，记录损失曲线，并从测试集采样音乐条件生成动作样例进行可视化，从而综合评估模型的收敛情况和生成质量。
import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4, # 新参数 = 旧参数 - 学习率 × 梯度
        weight_decay=0.02,
    ):
        # 初始化 accelerate(一个调度cpu和gpu的库) | 获取当前训练用了几个进程/GPU
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # 
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) # 
        state = AcceleratorState()
        num_processes = state.num_processes


        # 定义动作表示维度 repr_dim
        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        use_baseline_feats = feature_type == "baseline"
        feature_dim = 35 if use_baseline_feats else 4800 # Jukebox是4800维的隐藏层,Baseline只有35维

        # 音乐长度
        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None # 模型权重 存档
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"] # ?

        model = DanceDecoder(
            nfeats=repr_dim, # 每帧动作特征维度，151
            seq_len=horizon, # 序列长度(帧)
            latent_dim=512, # Transformer内部隐藏维度
            ff_size=1024, # 前馈层维度
            num_layers=8, # 8层网络
            num_heads=8, # 8头注意力
            dropout=0.1,
            cond_feature_dim=feature_dim, # 音乐维度，35 或 4800
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(self.accelerator.device)


        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine", # 扩散噪声调度表用 cosine ?  噪声增加的速度按照余弦曲线变化
            n_timestep=1000, # 扩散步数 1000
            predict_epsilon=False, # 不直接预测噪声 epsilon，而是预测别的目标（通常是 x0）
            loss_type="l2", # L2损失
            use_p2=False, # 不使用 p2 weighting
            cond_drop_prob=0.25, # 25% 概率丢弃条件， 训练时有时不给音乐条件，让模型同时学会：有条件生成\无条件生成 , 这样,可以使用guidance_weight增强音乐控制效果
            guidance_weight=2, # 生成时条件引导强度
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters())) # 打印模型参数总数，方便看网络规模
        )

        # 
        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 优化器
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        ###  load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        ###  set normalizer (把数据归一化,方便进行设置)
        self.normalizer = test_dataset.normalizer

        # data loaders CPU将数据传给GPU
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32), # 多开CPU进程
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)

        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )


        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(
                project=opt.wandb_pj_name, 
                name=opt.exp_name,
                config={
                    "lr": 4e-4, # 这里步长用的这个,在init里
                    "batch_size": opt.batch_size,
                    "epochs": opt.epochs
                },
                save_code = True
            )
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        # 所有进程在这里等一下，等主进程把目录、wandb 等准备好，再一起继续
        self.accelerator.wait_for_everyone()


        ### 正式开始训练
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0 # 主损失
            avg_vloss = 0 # 速度相关损失
            avg_fkloss = 0 # 前向运动学相关损失
            avg_footloss = 0 # 脚部接触/脚滑相关损失
            # train
            self.train() # 把模型设为训练模式
            for step, (x, cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion( # 前向传播,拿到各项损失
                    x, cond, t_override=None
                )
                self.optim.zero_grad() # 每次反向传播前都要先把上一步的梯度清空
                self.accelerator.backward(total_loss) # 反向传播计算梯度

                self.optim.step() # 根据梯度更新参数(用Adan优化器加速更新)

                # ema update and train loss update only on main # 把当前 batch 的各项损失累加起来，等这个 epoch 结束后求平均
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    # EMA = Exponential Moving Average，指数滑动平均。 # 它不是直接用当前训练模型参数，而是维护一个“更平滑”的参数版本 EMA 模型通常在生成任务中更稳定，因此在保存和采样时使用 EMA 权重
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average( # 训练过程中每隔若干 step，就用当前模型去更新 EMA 模型。
                            self.diffusion.master_model,self.diffusion.model # EMA模型, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval() # 
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    ###  generate a sample 生成动画序列
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    ### draw a music from the test dataset
                    (x, cond, filename, wavnames) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
