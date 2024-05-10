# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random

import torch
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class GuidedDiffusion(torch.nn.Module):
    #类的初始化方法，目的加载预训练的模型
    def __init__(self, args, config, device=None, model_dir='pretrained/guided_diffusion'):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        model_config = model_and_diffusion_defaults() #该函数实现获取模型和扩散模型的默认设置
        model_config.update(vars(self.config.model)) #更新模型配置，vars()函数返回对象的属性
        print(f'model_config: {model_config}') #打印模型配置信息
        model, diffusion = create_model_and_diffusion(**model_config)#双星号表示解压字典，并将参数传递给模型和扩散对象
        model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu')) #加载预训练的 Guided Diffusion 模型的参数
        model.requires_grad_(False).eval().to(self.device)#模型处于评估模式无需梯度更新
        #如果配置中指定了使用 FP16（半精度浮点数），则将模型的参数转换为 FP16 格式
        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)#把diffusion.betas转化为Numpy数组，再转化为浮点数类型。扩散模型中的超参数betas

    def image_editing_sample(self, img, bs_id=0, tag=None):#bs_id表示当前批次的标号
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)#断言输入的 img 必须是 PyTorch 的张量类型。
            batch_size = img.shape[0]#获取图片的批次大小

            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000)) #若标签为None,则生成一个随机标签,确保输出文件夹的唯一性
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)#输出路径

            assert img.ndim == 4, img.ndim#断言输入图片的维度必须为4
            img = img.to(self.device)
            x0 = img#作为原始的输入图像

            if bs_id < 2: #这里会输出批次0和1的图片
                os.makedirs(out_dir, exist_ok=True)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))
                #为什么要执行(x0 + 1) * 0.5操作？


            xs = []
            # sample_step =1 仅进行一次diffusion的前向过程，包括一个400步的加噪去噪过程
            for it in range(self.args.sample_step):
                e = torch.randn_like(x0) #e为形状与x0相同的张量，其值为从标准的正态分布中采样的随机值。作为扩散过程添加的噪声
                total_noise_levels = self.args.t #总的扩散步长
                a = (1 - self.betas).cumprod(dim=0) #计算的是扩散模型中的a一巴
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                #扩散模型前向加噪的过程

                ori_x = []
                #定义反向循环依次将将加噪的图片添加到列表ori_x[]中
                for t_tmp in reversed(range(0, total_noise_levels - 1)):
                    ori_x.append(x0 * a[t_tmp].sqrt() + e * (1.0 - a[t_tmp]).sqrt())
                ori_x.append(x0)

                if bs_id < 2:
                    tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

                mid_x = []
                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=self.device)
                    #p_sample()实现逐步去噪
                    x = self.diffusion.p_sample(self.model, x, t,
                                                clip_denoised=True,
                                                denoised_fn=None,
                                                cond_fn=None,
                                                model_kwargs=None)["sample"]

                    mid_x.append(x)

                    # added intermediate step vis
                    if (i - 99) % 100 == 0 and bs_id < 2:
                        tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'noise_t_{i}_{it}.png'))

                x0 = x

                if bs_id < 2:
                    torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

                xs.append(x0)

            return torch.cat(xs, dim=0), mid_x, ori_x
