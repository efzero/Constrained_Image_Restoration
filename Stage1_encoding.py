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
        default = "/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/checkpoints/sd1.5/2025-02-24-10-49-35_ucf101_2000.ckpt",
#         default="/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/checkpoints/sd1.2/2025-02-24-10-28-55_ucf101_2000.ckpt",
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

    
    df = pd.read_csv("/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/Training_metadata.csv")
    prefix = "/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/prior"
    revt = 49
    
    for i in range(df.shape[0]):
        sc_ = None
        
        hr_list = []
        img_name = df.loc[i]['image_name']
        prompt = df.loc[i]['prompt']
        prompts = [prompt]
        hr = cv2.imread(f"{prefix}/{img_name}")
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = np.transpose(hr,(2, 0, 1))/255
        hr = torch.tensor(hr)
        hr = (hr-hr.min())/(hr.max()-hr.min())
        hr_list.append(hr)
        x = torch.stack(hr_list, dim = 0)
        c = model.get_learned_conditioning(prompts)
        c = c.to(device)
        uc = model.get_learned_conditioning(batch_size * [""])
        uc = uc.to(device)
        x = x.to(device).to(model.first_stage_model.dtype)
        x = x * 2 -1
        z = model.get_first_stage_encoding(model.encode_first_stage(x))
        model.eval()
        code = sampler.encode_CIS(z, c, revt, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None, save_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/ddiminverse/{str(i).zfill(6)}.npy")
        if i % 20 == 0:
            print("sanity check")
            samples_ddim = sampler.decode(code, c, revt, unconditional_guidance_scale = 1.0, unconditional_conditioning = uc, sc = None)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            np.save(f"/scratch/liyues_root/liyues/shared_data/bowenbw/Constrained_Image_Restoration/data/ucf101/train/ddiminverse/sanity_check/{str(i).zfill(4)}_sanity.npy", x_samples_ddim)
        
        
if __name__ == "__main__":
    main()