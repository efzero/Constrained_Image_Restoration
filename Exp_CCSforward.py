import argparse, os, sys
from glob import glob

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
# from CT_dataloader import CTDataset
# from src.datasets import load_dataset_JIF
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torch.utils.data import Dataset, ConcatDataset
# from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from multiprocessing import Manager
# from src.load_dataset import *
# from src.datasources import S2_ALL_12BANDS
import random

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)   

now_ = datetime.now()
ymdh = now_.strftime("%Y-%m-%d %H-%M-%S")

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def sample_test(im_name_list, model, sampler, opt, sc_ = None, c = None, lr = None, ymdh = None, batch_size = 16):
        start_code = None
        with torch.no_grad():
            uc = None
            if opt.scale != 2.0:
                uc = model.get_learned_conditioning(batch_size * [""])

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=[c,lr],
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 sc = sc_,
                                                 x_T=start_code)

                x = model.decode_first_stage(samples_ddim)
                print(x.max(), x.min(), 'xmax, xmin')
                print(x.shape, 'x shape')
                xmin = x.view(x.size(0), -1).min(1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
                xmax = x.view(x.size(0), -1).max(1, keepdim=True)[0].view(x.size(0), 1, 1, 1)
                np.save('x_samples_ddim.npy', x.detach().cpu().numpy())
                x = (x - xmin)/(xmax - xmin)
                print(x.max(), x.min(), 'normalized x')
                # print(x.shape, 'x shape')
                x_samples_ddim = x.clone()
#                 x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                # np.save('x_samples_ddim.npy', x_samples_ddim)

                base_dir = "/scratch/liyues_root/liyues/shared_data/bowenbw/satellite_diffusion/samples/fmow"
                full_dir = os.path.join(base_dir, ymdh)
                if not os.path.exists(full_dir):
                    os.makedirs(full_dir)
                    
                for l in range(batch_size):
                    plt.imsave(full_dir + "/" + im_name_list[l] + ".png", (x_samples_ddim[l]*255).astype('uint8'))



def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downscale factor",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="ddim steps",
    )

    parser.add_argument(
        "--eta",
        type=int,
        default=0.0,
        help="randomness parameter",
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/nfs/turbo/coe-liyues/bowenbw/stable-diffusion/sd1.2/sd-v1-2.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        '--batch_size', 
        default=4, 
        type=int, 
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus',
    )

    opt = parser.parse_args()

#     seed_everything(opt.seed) ####unseed this for training

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    prompt = ["Satellite image"]
    iterations = 0
    batch_size = 1
    
    model.train()
    model.learning_rate = 1e-5
    model.use_scheduler = False
    
    opt_ = model.configure_optimizers()
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=True)
    
    
    
    sc_ = None
    
    
    hr_list = []
    hr_recon_list = []
    
    hr = cv2.imread("/scratch/liyues_root/liyues/shared_data/bowenbw/iclr2025satdiffmoe_data/trainingdata_new/airport/hr_00000.png")
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    hr = np.transpose(hr,(2, 0, 1))/255
    hr = torch.tensor(hr)
    hr = (hr-hr.min())/(hr.max()-hr.min())
    hr_list.append(hr)
    
    
    hr_recon = cv2.imread("/scratch/liyues_root/liyues/shared_data/bowenbw/iclr2025satdiffmoe_data/trainingdata_new/airport/hr_recon_00000.png")
    hr_recon =  cv2.cvtColor(hr_recon, cv2.COLOR_BGR2RGB)
    hr_recon = np.transpose(hr_recon,(2, 0, 1))/255
    hr_recon = torch.tensor(hr_recon)
    hr_recon = (hr_recon-hr_recon.min())/(hr_recon.max()-hr_recon.min())
    hr_recon_list.append(hr_recon)
    
    
    lr_list = []
    lr = cv2.imread('/scratch/liyues_root/liyues/shared_data/bowenbw/iclr2025satdiffmoe_data/trainingdata_new/airport/lr_00000_0.png')
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    lr = np.transpose(lr,(2, 0, 1))/255
    lr = torch.tensor(lr)
    lr = (lr-lr.min())/(lr.max()-lr.min())
    lr_list.append(lr)
    
    x = torch.stack(hr_list, dim = 0)
    x_recon = torch.stack(hr_recon_list, dim = 0)
    y = torch.stack(lr_list, dim = 0)
    
    prompt = ["satellite image"]
    c = model.get_learned_conditioning(batch_size*prompt)
    c = c.to(device)
    
    uc = model.get_learned_conditioning(batch_size * [""])
    uc = uc.to(device)
    
    x = x.to(device).to(model.first_stage_model.dtype)
    x_recon = x_recon.to(device).to(model.first_stage_model.dtype)
    y = y.to(device).to(model.first_stage_model.dtype)
    xmax_, ymax_ = x.max(), y.max()
            
    x = x * 2 -1
    x_recon = x_recon * 2 - 1
    y = y * 2 -1 
    z = model.get_first_stage_encoding(model.encode_first_stage(x))
    z1 = model.get_first_stage_encoding(model.encode_first_stage(x_recon))
    z2 = model.get_first_stage_encoding(model.encode_first_stage(y))
    
    
    
    model.train()
    for training_itrs in range(20):
        sc_ = None
        loss,loss_dict = model.forward(z, c, sc = None)
        loss.backward()
        opt_.step()
        opt_.zero_grad()
            
            
    
#     cc = c 
    model.eval()
#     t = torch.randint(0, 10, (x.shape[0],), device=device).long()
    
    revt = 49
    
    alpha_t = sampler.model.alphas_cumprod[revt * 20]
    clean_coef = torch.sqrt(alpha_t)
    noise_coef = torch.sqrt(1 - alpha_t)
    
    encoded = sampler.encode(z, c, revt, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None)
    
#     encoded_noise = encoded - clean_coef * z
#     new_zt = clean_coef * (z1 - z) + encoded
#     new_zt = encoded

    noise = encoded - clean_coef * z
    noise = (noise - noise.mean())/noise.std()
    
    new_zt = clean_coef * z1 + noise_coef * noise
    sample_im = sampler.decode(new_zt, c, revt, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None)
    x = model.decode_first_stage(sample_im)
    im = torch.clamp((x+1)/2, 0, 1)
    
    
    np.save("Exp_CCSforward_0.0_revt50.npy", im.cpu().numpy())
    
        
        
        
        
#     model.model.diffusion_model(z, t, context=cc)
    
    
    
            
            
if __name__ == "__main__":
    main()