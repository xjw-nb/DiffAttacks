import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from ddpm.unet_cifar_ddpm import UNet

import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))




class Diffusion_cifar(torch.nn.Module):
    def __init__(self, args, device=None):
        super().__init__()

        self.args = args

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        print("Loading model")
        model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],num_res_blocks=2, dropout=0.1)
        ckpt = torch.load('./pretrained/ckpt.pt')
        model.load_state_dict(ckpt['net_model'])
        self.model = model.to(device)


        # args of ddpm
        mean_type = 'epsilon'
        var_type = 'fixedlarge'
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        self.T = 1000
        self.img_size = 32
        self.mean_type = mean_type
        self.var_type = var_type
        beta_1 = 0.0001
        beta_T = 0.02
        self.sample_step=1

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        #model_log_var为一个字典，根据var_type选择不同的模型对数方差
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            #torch.cat表示对两部分进行拼接
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

    def image_editing_sample(self, img=None, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        if self.args.t == 0:
            self.mid_x = [img]
            self.ori_x = [img]
            self.total_noise_levels = self.args.t
            return img, self.mid_x, self.ori_x

        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        # out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        assert img.ndim == 4, img.ndim
        x0 = img

        # if bs_id < 2:
        #     os.makedirs(out_dir, exist_ok=True)
        #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

        xs = []

        self.noises = []

        with torch.no_grad():
        # with torch.enable_grad():

            for it in range(self.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0).to(x0.device)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                self.e = e
                self.a = a
                self.total_noise_levels = total_noise_levels

                ori_x = []
                for t_tmp in reversed(range(0, total_noise_levels)):
                    ori_x.append(x0 * a[t_tmp].sqrt() + e * (1.0 - a[t_tmp]).sqrt())
                ori_x.append(x0)

                self.ori_x = ori_x

                # if bs_id < 2:
                #     tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

                mid_x = []
                mid_x.append(ori_x[0])
                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=img.device)

                    x, log_var = self.p_mean_variance(x_t=x, t=t)
                    # no noise when t == 0
                    if i > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = 0

                    self.noises.append(noise)

                    x = x + torch.exp(0.5 * log_var) * noise

                    mid_x.append(x)



                x0 = x
                x0 = torch.clip(x0, -1, 1)

                # if bs_id < 2:
                #     torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

                xs.append(x0)

        self.mid_x = mid_x

        return torch.cat(xs, dim=0), mid_x, ori_x

    # grad=∂L/∂x't-1  ,函数目的:求出∂L/∂x't 并释放∂L/∂x't-1的内存
    #假设时间步为400，mid_x[idx]=[x'399,x'398 .....x'1,x'0],反向去噪的样本
    def compute_efficient_gradient(self, grad, idx):
        batch_size = self.mid_x[idx].shape[0] #mid_x[idx]表示一个张量，.shape[0]表示该张量的第一个维度。idx表示扩散步数的一个索引

        with torch.enable_grad(): #表示下面代码需要用梯度计算
            # self.mid_x[idx] = torch.tensor(self.mid_x[idx], requires_grad=True, device=self.mid_x[idx].device)
            self.mid_x[idx] = self.mid_x[idx].clone().detach().requires_grad_(True)
            #total_noise_levels，表示总的噪声采样的步数400
            t = torch.tensor([self.total_noise_levels-1-idx] * batch_size, device=self.mid_x[idx].device)
            #张量t不知道什么作用
            x, log_var = self.p_mean_variance(x_t=self.mid_x[idx], t=t)
            #p_mean_variance()计算了条件概率分布 pθ(x'0|x'1) 的均值 x 和方差 log_var
            noise = self.noises[idx] #获得当前索引的噪声
            x = x + torch.exp(0.5 * log_var) * noise #对均值x的值进行调整


            loss = torch.sum(x * grad)

            # print(f'x.shape: {x.shape}')
            # print(f'grad.shape: {grad.shape}')
            # print(f'loss.shape: {loss.shape}')

            grad_new = torch.autograd.grad(loss, [self.mid_x[idx]])[0].detach()

        return grad_new

    #classifier被攻击的分类器，loss_funct为交叉熵损失函数，x为输入图片x'0,label为真实标签
    #该函数用于求出损失对输入图片x'0的梯度
    def compute_efficient_gradient_classifier(self, classifier, loss_funct, x, label):
        with torch.enable_grad():
            # x = torch.tensor(x,requires_grad=True,device=label.device)
            x = x.clone().detach().requires_grad_(True)

            out = classifier(x)
            loss__ = loss_funct(out, label) #求预测标签与真实标签的一个损失
            loss_ce = loss__.clone()
            loss = torch.mean(loss__)
            #计算标量损失对对张量x的梯度
            grad_new = torch.autograd.grad(loss, [x])[0].detach() #torch.autograd.grad计算loss对变量x的梯度。[0]表示返回梯度列表的第一个元素

        return loss_ce, grad_new

    def compute_efficient_gradient_diffusion_process(self, ori_x, grad):
        with torch.enable_grad():
            if self.total_noise_levels >= 1:
                out = ori_x * self.a[self.total_noise_levels - 1].sqrt() + self.e * (1.0 - self.a[self.total_noise_levels - 1]).sqrt()

                loss = torch.sum(out * grad)

                grad_new = torch.autograd.grad(loss, [ori_x])[0].detach()
            else:
                grad_new = grad

        return grad_new

    # ori_x=[ xT  xT-1 .... x1 x0 ] x0到xT是逐步加噪的图片
    # mid_x=[XT x'T-1 x'T-2 ......x'1  x'0] xT到x0的逐步去噪图片，idx是下标，从0开始
    # grad=∂L/∂x't-1  ,函数目的:求出∂L/∂x't 并释放∂L/∂x't-1的内存
    # 和上一个函数的区别是这个往原先的交叉熵损失加多了一个偏差重建损失，偏差重建损失仅添加在部分时间步
    def compute_efficient_gradient_mse(self, grad, idx): #偏差重建损失
        batch_size = self.mid_x[idx].shape[0]

        with torch.enable_grad():
            # self.mid_x[idx] = torch.tensor(self.mid_x[idx], requires_grad=True, device=self.mid_x[idx].device)
            self.mid_x[idx] = self.mid_x[idx].clone().detach().requires_grad_(True)

            t = torch.tensor([self.total_noise_levels-1-idx] * batch_size, device=self.mid_x[idx].device)

            x, log_var = self.p_mean_variance(x_t=self.mid_x[idx], t=t) #x_t当前时间步的样本，
            noise = self.noises[idx]
            x = x + torch.exp(0.5 * log_var) * noise

            loss = torch.sum(x * grad)

            mse = torch.nn.MSELoss(reduction='none')
            mid_x = self.mid_x[idx]
            ori_x = self.ori_x[idx]
            # || x't - xt ||2 偏差重建损失
            # 计算两个张量 mid_x 和 ori_x 之间的均方误差 (MSE) 损失
            #.view()对 ori_x 张量进行形状调整，使其形状与 mid_x 相同
            loss_mse = mse(mid_x, ori_x.view(mid_x.shape))
            loss_mse = loss_mse.view(loss_mse.shape[0],-1) #将loss_mse的形状进行重塑为(loss_mse.shape[0],-1)。-1表示改维度的大小由数据自行推断
            loss_mse = torch.mean(loss_mse, dim=-1)
            loss_mse = torch.mean(loss_mse, dim=0)


            loss = loss + loss_mse


            grad_new = torch.autograd.grad(loss, [self.mid_x[idx]])[0].detach()

        return grad_new