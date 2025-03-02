"""SAMPLING ONLY."""

import torch
import numpy as np
import lpips
from tqdm import tqdm
from functools import partial
from matplotlib import cm
import matplotlib.pyplot as plt
import random
from einops import rearrange, repeat
from torchvision import transforms
from PIL import Image
import wandb
# from blocks import *
import numpy as np



from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.null_cond = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.model.betas.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))
    
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               sc = None,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, vps = False, resample = False, blend_param = 0.0, pixelavg = False, y = None, CompareC = False, scoreedit = False, good_indices = None, fusion = False, cfg_cutoff = 50, 
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                cbs = conditioning[0].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size, sc = sc,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning, vps = vps, resample = resample, pixelavg = pixelavg, blend_param = blend_param, y = y, CompareC = CompareC, scoreedit = scoreedit, good_indices = good_indices, fusion = fusion, cfg_cutoff= cfg_cutoff)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, sc = None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,vps = False, resample = False, pixelavg = False, blend_param = 0.0, y = None, CompareC = False, rblend = False, scoreedit = False, good_indices = None, fusion = False, cfg_cutoff=50):
        
        print(cond[0].shape, "cond shape")
        device = self.model.betas.device
        
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        print(self.model, file=open('modellog.txt', 'a'))
        
        loss_fn_lpips = lpips.LPIPS(net='vgg',version='0.1').cuda()
        
#         drop_proba = np.random.rand()
#         if drop_proba < 0.8:
#             print(c[1].shape)
#             c[1] = c[1][:,:4,:,:]
        
        ########################load the measurement y##########################################
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            new_cond = cond
            if rblend:
                new_cond0 = cond[0][torch.randperm(cond[0].size(0))]
                new_cond = [new_cond0, cond[1]]

            outs = self.p_sample_ddim(img, new_cond, ts, index=index, sc = sc, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, vps = vps, resample = resample, pixelavg = pixelavg, blend_param = blend_param, y = y, scoreedit = scoreedit, good_indices = good_indices, fusion = fusion, lpips_loss = loss_fn_lpips, cfg_cutoff=cfg_cutoff)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        if CompareC:
            t = random.randint(0, 50)
            t = repeat(torch.tensor([t]), '1 -> b', b = 16)
            t = t.to(device).long()
            x, noise = self.stochastic_encode(img, t)
            ########################################Find Conditions and ApplyModel to estimate noise###################################
            c = new_cond
            b, *_, device = *x.shape, x.device
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c, sc = sc, y = y) ###sc: z, z+1, z+2, z+3
                ##########################testing blending################################
                cond_score = torch.zeros_like(e_t).to(device)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                new_c = None
                if type(c) == list:
                ################################lr image as a new condition################################
                    c_in = torch.cat([unconditional_conditioning, c[0]]) 
                    lr_in = torch.cat([c[1], c[1]])
                    new_c = [c_in, lr_in]
                else: #######################only with sentence embedding
                    c_in = torch.cat([unconditional_conditioning, c])
                    new_c = c_in
    
                sc_in = None
                if sc is not None:
                    sc_in = torch.cat([sc] * 2)
                    
                y_in = None
                if y is not None:
                    y_in = torch.cat([y] * 2)
                # e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, sc = sc_in).chunk(2)  #####for only prompt conditioning
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, new_c, sc = sc_in, y = y_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            ##################################################END###############################################
            error = e_t_uncond - noise #####Bowen Changed to unconditional
            norm_error = torch.mean(error**2, dim=(1,2,3))
            _, idx = torch.min(norm_error, dim = 0)

        return img, intermediates

    @torch.no_grad()  ###disable this for inverse problem solving
    def p_sample_ddim(self, x, c, t, index = None, sc = None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, vps = False, resample = False, pixelavg = False, x_true = None, ind = None, blend_param = 0.0, y = None, scoreedit = False, good_indices = None, fusion = False, lpips_loss = None, key = 'default', HR_prior = None, lr = 1e-3, cfg_cutoff = 25,hr = None, model1 = None, preprocess = None, model_alpha = None):
        
        b, *_, device = *x.shape, x.device

        cond = None
        if type(c) == list:
            drop_proba = np.random.rand()
            if drop_proba > 1:
                cond = c[1][:,:4,:,:]
            else:
                cond = c[1]
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, sc = sc, y = y, key = key) ###sc: z, z+1, z+2, z+3
            ##########################testing blending################################
            cond_score = torch.zeros_like(e_t).to(device)
        else:
            x = x.detach()
            x.requires_grad = True
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            new_c = None
            if type(c) == list:
            ################################lr image as a new condition################################
                c_in = torch.cat([unconditional_conditioning, c[0]]) ######text embeddings
                lr_in = torch.cat([cond, cond]) #####c[1] is lr #########latent conditions
                print(lr_in.shape, "lr in shape", cond.shape, "cond shape")
                new_c = [c_in, lr_in]
            else: #######################only with sentence embedding
                c_in = torch.cat([unconditional_conditioning, c])
                new_c = c_in
            sc_in = None
            if sc is not None:
                sc_in = torch.cat([sc] * 2)
            y_in = None
            if y is not None:
                y_in = torch.cat([y] * 2)
            
                
            # e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, sc = sc_in).chunk(2)  #####for only prompt conditioning
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, new_c, sc = sc_in, y = y_in, key = key).chunk(2)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
            ###########################################################################################
        ##########################################Compute predicted z0###############################################
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        print(a_t[0].item(), index, t[0].item(), file = open("atlog.txt", "a"))
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)  
        pred_z0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        norm_grad = None
        if HR_prior is not None:
            print("computing norm grad")
            diff = HR_prior - pred_z0
            norm = torch.linalg.norm(diff)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            print("norm grad shape", norm_grad.shape)
            
        x = x.detach()
        projected_z0 = pred_z0


        e_t = e_t
        pred_x0 = pred_z0
        x0_pred = self.model.decode_first_stage(pred_x0) ###1 x 3 x 512 x 512
        tensor = x0_pred
        tensor = (tensor + 1) * 0.5 * 255.0
        tensor = tensor.clamp(0, 255)  # Ensure values are within [0, 255]
        # Convert to NumPy array

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        if norm_grad is not None:
            x_prev = x_prev - 5 * a_t.sqrt() * norm_grad
        return x_prev.detach(), pred_x0
    
    @torch.no_grad()
    def encode(self, x0, c, t_enc, sc = None, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None, rho = 0.1, save_path = ""):
        

        rho = 0.0
        if type(c) != list:
            self.model.conditioning_key = 'crossattn'
            
        if type(c) == list:
            ################################lr image as a new condition################################
            c_in = torch.cat([unconditional_conditioning, c[0]]) 
            lr_in = torch.cat([c[1], c[1]])
            new_c = [c_in, lr_in]
            self.model.conditioning_key = 'hybrid'
#             self.model.conditioning_key = 'priorfusion_new'
            print("loading new c")
            
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]
        

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                ############compute score###################################
                sc_in = sc
                noise_pred = self.model.apply_model(x_next, t, c, sc=sc_in, key = self.model.conditioning_key)
            else:
                assert unconditional_conditioning is not None
                ############compute score###################################
                sc_in = None
                if sc is not None:
                    sc_in = torch.cat([sc] * 2)
#                 e_t_uncond, noise_pred = torch.chunk(
#                     self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
#                                            torch.cat((unconditional_conditioning, c)), sc = sc_in), 2)  ########without LR reverse sampling
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)), torch.cat((unconditional_conditioning, c)), sc = sc_in), 2) #####with LR inversion
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            
            x_orig = x_next.clone()
            x_next = xt_weighted + weighted_noise_pred ####\hat{z}_t
            
            ################################Exact Inversion##########################################
#             e_t_uncond, noise_pred = torch.chunk(self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
#                                            new_c, sc = sc_in), 2)
#             noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond) ######new score

#             pred_x0 = (x_next - (1 - alphas_next[i]).sqrt() * noise_pred) / alphas_next[i].sqrt()
#         # direction pointing to x_t
#             dir_xt = (1. - alphas[i]).sqrt() * noise_pred
#             x_resample = alphas[i].sqrt() * pred_x0 + dir_xt  ####z'{t-1}
#             #################################forward step########################## z_t-1 -= rho (x_resample - x_orig)
#             x_next -= rho * (x_resample - x_orig)  ####Update
            ##############################End Exact Inversion##########################################
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)
#             file_name = f'{save_path}_{str(i).zfill(3)}.npy'
#             np.save(file_name, x_next.detach().cpu().numpy())

#         out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
#         if return_intermediates:
#             out.update({'intermediates': intermediates})
        return x_next



    @torch.no_grad()
    def encode_CIS(self, x0, c, t_enc, sc = None, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None, rho = 0.1, save_path = ""):
        
#         print(type(c), "start encoding")

        rho = 0.0
        if type(c) != list:
            self.model.conditioning_key = 'crossattn'
            
        if type(c) == list:
            ################################lr image as a new condition################################
            c_in = torch.cat([unconditional_conditioning, c[0]]) 
            lr_in = torch.cat([c[1], c[1]])
            new_c = [c_in, lr_in]
            self.model.conditioning_key = 'hybrid'
#             self.model.conditioning_key = 'priorfusion_new'
            print("loading new c")
            
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]
        

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = [x0]
        inter_steps = []
        
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                ############compute score###################################
                sc_in = sc
                noise_pred = self.model.apply_model(x_next, t, c, sc=sc_in, key = self.model.conditioning_key)
            else:
                assert unconditional_conditioning is not None
                ############compute score###################################
                sc_in = None
                if sc is not None:
                    sc_in = torch.cat([sc] * 2)
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)), torch.cat((unconditional_conditioning, c)), sc = sc_in), 2) #####with LR inversion
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            
            x_orig = x_next.clone()
            x_next = xt_weighted + weighted_noise_pred ####\hat{z}_t
            intermediates.append(x_next)
            
            ##############################End Exact Inversion##########################################
#             if return_intermediates and i % (
#                     num_steps // return_intermediates) == 0 and i < num_steps - 1:
#                 intermediates.append(x_next)
#                 inter_steps.append(i)
#             elif return_intermediates and i >= num_steps - 2:
#                 intermediates.append(x_next)
#                 inter_steps.append(i)
#             if callback: callback(i)
#             file_name = f'{save_path}_{str(i).zfill(3)}.npy'
#             np.save(file_name, x_next.detach().cpu().numpy())

#         out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
#         if return_intermediates:
#             out.update({'intermediates': intermediates})

        if save_path != "":
            intermediates_arr = torch.stack(intermediates).detach().cpu().numpy()
            np.save(save_path, intermediates_arr)
        return x_next

    
    

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        
        print(use_original_steps, "use original steps")
        print(t.shape, x0.shape,  "t", "x0")
        
        use_original_steps = False
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
            
        print((extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape)), "at", extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape), "1-at")

#         if noise is None:
#             noise = torch.randn_like(x0[1]).repeat(x0.shape[0],1,1,1) ###1, 64, 64 ###old
        if noise is None:
            noise = torch.randn_like(x0).to(x0.device)

        device = x0.device
#         print(t.dtype)
        t = t.to(torch.int64)

        print((extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape)), "at", extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape), "1-at")
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise), noise

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, sc = None, lr = 1e-3, cfg_cutoff = 25, HR_prior = None):
        """
        takes:
        1. t_start: start point of noise
        2. cond: conditional information
        3. hr: prior image
        
        returns: 
        1. decoded
        
        
        """
        """
        
        img, new_cond, ts, index=index, sc = sc, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, vps = vps, resample = resample, pixelavg = pixelavg, blend_param = blend_param, y = y
                                      """
        if type(cond) != list:
            self.model.conditioning_key = 'crossattn'
        else:
            self.model.conditioning_key = 'hybrid'
#             self.model.conditioning_key = 'priorfusion_new'

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        print(x_dec.shape, "xdec")

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
#             print(step, index)
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning, sc = sc, y = None, resample = False, vps = False, key = self.model.conditioning_key, lr = lr, cfg_cutoff = cfg_cutoff, HR_prior = HR_prior, model1 = None, preprocess = None, model_alpha = None)
        return x_dec
    
    
    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-4, max_iters=300,true_img = None):
    
        ###steps: 300 for sr, 500 for nonlinear, gauss deblur, 200 for inpaint
        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        # Base case
        if not z_init.requires_grad:
            print("added gradient")
            z_init = z_init.requires_grad_()
        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=5e-3) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons
        # Training loop
        prev_loss = -1
        init_loss = 0       
        
        cum_loss = 0
        losses = []
        
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn(self.model.differentiable_decode_first_stage( z_init ) ))          
            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy()
            if itr %10 == 0:
                print(cur_loss, itr)
                if true_img is not None:
                    deblur_psnr = psnr(clear_color(true_img), clear_color(self.model.decode_first_stage(z_init)))
                    print(deblur_psnr, itr)    
        #####detecting gradient descent converging/overfitting ############
            if itr < 200:
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)                

        return z_init, init_loss
    
   