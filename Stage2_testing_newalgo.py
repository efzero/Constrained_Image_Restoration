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
import pandas as pd
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import torch.nn.functional as F

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)   

now_ = datetime.now()
ymdh = now_.strftime("%Y-%m-%d-%H-%M-%S")

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

def cosine_similarity_images(image1, image2):

    batch1_flat = image1.view(image1.size(0), -1)
    batch2_flat = image2.view(image2.size(0), -1)

    cosine_sim = F.cosine_similarity(batch1_flat, batch2_flat, dim=1)
    angles = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0))
    return angles

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
#         default="/nfs/turbo/coe-liyues/bowenbw/stable-diffusion/sd1.2/sd-v1-2.ckpt",
#         default="/nfs/turbo/coe-liyues/bowenbw/stable-diffusion/v1-5-pruned.ckpt",
        default="/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/checkpoints/sd1.5/2025-02-24-10-49-35_ucf101_2000.ckpt",
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

    iterations = 1
    
    df = pd.read_csv("/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/Training_metadata.csv")
    prefix = "/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/cur"
    prefix_prior = "/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/prior"
    
    
    
    randseed = None
    
    for itrs in range(iterations):
        
        sc_ = None
        hr_list = []
        selected_im_inds = [0]
        
        hr_prior_list = []
        
        batch_train_imgs = list(df.loc[selected_im_inds]['image_name'])
        batch_train_prompts = list(df.loc[selected_im_inds]['prompt'])
        
        
        for i in range(batch_size):
            hr = cv2.imread(f"{prefix}/{batch_train_imgs[i]}")
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            hr = np.transpose(hr,(2, 0, 1))/255
            hr = torch.tensor(hr)
            hr = (hr-hr.min())/(hr.max()-hr.min())
            hr_list.append(hr)

        for i in range(batch_size):
            hr_prior = cv2.imread(f"{prefix_prior}/{batch_train_imgs[i]}")
            hr_prior = cv2.cvtColor(hr_prior, cv2.COLOR_BGR2RGB)
            hr_prior = np.transpose(hr_prior, (2,0,1))/255
            hr_prior = torch.tensor(hr_prior)
            hr_prior = (hr_prior - hr_prior.min())/(hr_prior.max() - hr_prior.min())
            hr_prior_list.append(hr_prior)
            
            
        x = torch.stack(hr_list, dim = 0)
        x_prior = torch.stack(hr_prior_list, dim = 0)
        
        c = model.get_learned_conditioning(batch_train_prompts)
        c = c.to(device)
        uc = model.get_learned_conditioning(batch_size * [""])
        uc = uc.to(device)
        
        
        x = x.to(device).to(model.first_stage_model.dtype)
        x_prior = x_prior.to(device).to(model.first_stage_model.dtype)
        
        
        x = x * 2 -1
        x_prior = x_prior * 2 -1
        
        z = model.get_first_stage_encoding(model.encode_first_stage(x))
        z_prior = model.get_first_stage_encoding(model.encode_first_stage(x_prior))
        
        model.train()
        
        sc_ = None
        
#         t = torch.randint(0, 50, (x.shape[0],), device=device).long()


        t = torch.randint(49, 50, (x.shape[0],), device=device).long()
        alpha_t = sampler.model.alphas_cumprod[t * 20]
        clean_coef = torch.sqrt(alpha_t)
        noise_coef = torch.sqrt(1 - alpha_t)
        ddim_prefix = "/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/ddiminverse/"
        ddim_invs = torch.from_numpy(np.load(f"{ddim_prefix}{str(0).zfill(6)}.npy")).to(device)
        code = ddim_invs[-1]
        print(code.shape, "code shape")
        rand_noise = torch.randn_like(code).to(device) ###N(0,1)
        
#         if randseed is None:
#             randseed = torch.randn_like(code).to(device)
#         code = code / noise_coef ###normalized to N(0,1)

        
        #############################################debugging#################################################
        
        #####################################sequential fine tuning############################################
#         code = randseed
        
        theta = cosine_similarity_images(code, rand_noise)
        theta = theta.view(batch_size, 1,1,1)
        C0 = 0.0
        C0test = 0.0
        C_ = torch.tensor(C0).to(z.device)
        C_test = torch.tensor(C0test).to(z.device)
#         t = 49
        
        
        #################################################initialization############################
        start_noise = torch.sin(C_)/torch.sin(theta) * rand_noise + torch.sin(theta - C_) / torch.sin(theta) * code
        gt_noise = start_noise
#         xt = clean_coef * z + noise_coef * start_noise
        xt = start_noise
        
        
        xt = sampler.decode(start_noise, c, 49, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None)
        

        
#         for k in range(50):
# #             loss, loss_dict = model.forward(z, c, t = t, xt = xt, sc = None, noise = gt_noise)
# #             loss.backward()
# #             opt_.step()
# #             opt_.zero_grad()
#             print("sampling at t", t)
#             xt, pred_x0 = sampler.p_sample_ddim(xt, c, t, index = t[0].item(), unconditional_guidance_scale=1.0,
#                                           unconditional_conditioning= uc, sc = None) #####reverse sample one step
#             xt = xt.detach()
#             ####updating

#             gt_noise = (xt - clean_coef * z) / noise_coef ####noise for xt-1
#             t = t - 1 ###t for xt-1
#             alpha_t = sampler.model.alphas_cumprod[t * 20] #### coefs for xt-1
#             clean_coef = torch.sqrt(alpha_t) #### coefs for xt-1
#             noise_coef = torch.sqrt(1 - alpha_t) #### coefs for xt-1
            
            
        x_samples_ddim = model.decode_first_stage(xt)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        np.save("DDIMsampletest.npy", x_samples_ddim)
            
            
        

        
        
        
        
        
            
#         final_noise = torch.sin(C_)/torch.sin(theta) * rand_noise + torch.sin(theta - C_) / torch.sin(theta) * code ###spherical interpolation
        
#         print(z.shape, c.shape, final_noise.shape, "z c noise")
        
        
        ####################################################################################################
            
#         if itrs % 10 == 0:
            
#             ddim_start = rand_noise #####testing generation capability
            
#             print("cur iteration is: ", iterations)
#             shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            
#             samples_ddim = sampler.decode(ddim_start, c, 49, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None)
#             x_samples_ddim = model.decode_first_stage(samples_ddim)
#             x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
#             x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
#             np.save(f"/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/Stage2_intermediates{itrs}_CIStraining0.4_increasingC_rand_partial.npy", x_samples_ddim)
            

#             start_code=ddim_invs[-1]
#             theta2 = cosine_similarity_images(start_code, rand_noise)
#             theta2 = theta2.view(batch_size, 1,1,1)
#             ddim_start = torch.sin(C_test) / torch.sin(theta2) * rand_noise + torch.sin(theta2 - C_test) / torch.sin(theta2) * start_code
# #             ddim_start = torch.clamp(ddim_start, -3.5, )
            
#             samples_ddim = sampler.decode(ddim_start, c, 49, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None)
#             x_samples_ddim = model.decode_first_stage(samples_ddim)
#             x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
#             x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
#             np.save(f"/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/Stage2_intermediates{itrs}_CIStraining0.4_increasingC_partial.npy", x_samples_ddim)
            
            
            
if __name__ == "__main__":
    main()