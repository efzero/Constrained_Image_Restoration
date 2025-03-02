"""SAMPLING ONLY."""

import torch
import numpy as np
import lpips
from tqdm import tqdm
from functools import partial
from test_CG import project
from matplotlib import cm
import matplotlib.pyplot as plt
from physics.ct import CT
import random
from einops import rearrange, repeat


from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

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
    
#     @torch.no_grad()
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
               unconditional_conditioning=None, vps = False, resample = False, blend_param = 0.0, pixelavg = False, y = None, CompareC = False, scoreedit = False, good_indices = None, fusion = False,
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
                                                    unconditional_conditioning=unconditional_conditioning, vps = vps, resample = resample, pixelavg = pixelavg, blend_param = blend_param, y = y, CompareC = CompareC, scoreedit = scoreedit, good_indices = good_indices, fusion = fusion)
        return samples, intermediates

#     @torch.no_grad()
    def ddim_sampling(self, cond, shape, sc = None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,vps = False, resample = False, pixelavg = False, blend_param = 0.0, y = None, CompareC = False, rblend = False, scoreedit = False, good_indices = None, fusion = False):
        
        
        print(cond[0].shape, "cond shape")
        device = self.model.betas.device
        
        
        loss_fn_lpips = lpips.LPIPS(net='vgg',version='0.1')
        loss_fn_lpips.cuda()
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
        ########################load the measurement y##########################################
#         img_np = np.load("ct3d_gt_test.npy")[:,0:1,:,:]
#         x_true = torch.from_numpy(img_np).cuda().to(torch.float)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
#             print(type(self.model))
#             print(cond[0].shape, "cond0shape")

            new_cond = cond
            if rblend:
                new_cond0 = cond[0][torch.randperm(cond[0].size(0))]
                new_cond = [new_cond0, cond[1]]
#             lr_per
#             new_cond.append(
            outs = self.p_sample_ddim(img, new_cond, ts, index=index, sc = sc, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, vps = vps, resample = resample, pixelavg = pixelavg, blend_param = blend_param, y = y, scoreedit = scoreedit, good_indices = good_indices, fusion = fusion, lpips_loss = loss_fn_lpips)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        if CompareC:
            t = random.randint(0, 49)
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

        # e_t: 16 * 4 * 24 * 24
#             error = e_t - noise ######Zhaoxu original
            error = e_t_uncond - noise #####Bowen Changed to unconditional
            norm_error = torch.mean(error**2, dim=(1,2,3))
            _, idx = torch.min(norm_error, dim = 0)
            # print(idx, file = open('/nfs/turbo/coe-liyues/luozhx/satellite_diffusion/compare/uc_idxes.txt', 'a') )
            # print(norm_error, file = open('/nfs/turbo/coe-liyues/luozhx/satellite_diffusion/compare/uc_errors.txt', 'a'))
            # img = img[idx][None]

        return img, intermediates

#     @torch.no_grad()  ###disable this for inverse problem solving
    def p_sample_ddim(self, x, c, t, index = None, sc = None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, vps = False, resample = False, pixelavg = False, x_true = None, ind = None, blend_param = 0.0, y = None, scoreedit = False, good_indices = None, fusion = False, lpips_loss = None):
        b, *_, device = *x.shape, x.device
        
        monte_carlo_inds = random.sample(range(0,16),10)
        lam = 0.1
        
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
            if blend_param > 0:
            ##########################testing blending################################
                cond_score = torch.zeros_like(e_t).to(device)
                e_t_uncond1, e_t1 = self.model.apply_model(x_in, t_in, c_in, sc = sc_in - 1, y = y_in).chunk(2) #z-1, z, z+1, z+2
                e_t1 = e_t_uncond1 + unconditional_guidance_scale * (e_t1 - e_t_uncond1)
                cond_score[:3] = (e_t1[1:] -e_t[:3]) * blend_param ####logp(x|y) - logp(x)
                e_t = e_t + cond_score
            ###########################################################################################
            
        ##########################################Compute predicted z0###############################################
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)  
        pred_z0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        projected_z0 = pred_z0
        pred_x0 = self.model.decode_first_stage(pred_z0)
        pred_x0_np = pred_x0.detach().cpu().numpy()
        pred_x0_np = (pred_x0_np - pred_x0_np.min())/(pred_x0_np.max() - pred_x0_np.min())
        t_np = int(t.detach().cpu().numpy()[0])
        
        ##########################################Post-processing######################################################    
        if scoreedit:
            e_t_mean = e_t[random.sample(good_indices, min(10, len(good_indices)))].mean(axis=0)
            direction = e_t - e_t_mean
            e_t = e_t - lam * direction
            
            ############################################################################################################

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        ########################################compute the predicted x0#####################################
#         pred_z0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt() ####16 x 4 x 24 x 24
#         pred_x0 = self.model.decode_first_stage(pred_z0)
#         pred_x0_np = pred_x0.detach().cpu().numpy()
#         pred_x0_np = (pred_x0_np - pred_x0_np.min())/(pred_x0_np.max() - pred_x0_np.min())
#         t_np = int(t.detach().cpu().numpy()[0])
        # np.save("/nfs/turbo/coe-liyues/luozhx/satellite_diffusion/samples/intermediate_samples/visit1_fixnoise/predx0_time" + str(t_np)+".npy", pred_x0_np)
        
        # for m in range(4):
            
            # plt.imsave("/nfs/turbo/coe-liyues/luozhx/satellite_diffusion/samples/intermediate_samples/visit1_fixnoise/predx0_time" + str(t_np) + "img_num" + str(m) + ".png", np.transpose(pred_x0_np[0], (1,2,0)))
        
        ######################################################################################################
        
        if resample and pixelavg:
            pred_x0 = self.model.decode_first_stage(pred_z0)
            if index % 5 != 2:
                print("pass", index)
            else:
                print("ReSampling!")
                lam = 0.1
#                 pred_x0_avg = pred_x0[monte_carlo_inds].mean(axis=0) #######original without filtering
                # good_indices = range(16)
                pred_x0_avg = pred_x0[random.sample(good_indices, min(10, len(good_indices)))].mean(axis=0)
                direction = pred_x0_avg - pred_x0
                projected_x0 = pred_x0 + lam * direction ####hopefully remain the original shape
                print(projected_x0.shape)
                sigma = 5*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                projected_z0 = self.model.get_first_stage_encoding(self.model.encode_first_stage(projected_x0)) ####original code by Zhaoxu
#         projected_z0 = self.model.encode_first_stage(projected_x0).mode() ####changed by Bowen to remove the stochasticity
                x = self.stochastic_resample(pseudo_x0=projected_z0, x_t=x, a_t=a_t, sigma=sigma)
                
                
        if resample and not pixelavg:
            if index % 5 != 2:
                print("pass", index)
            else:
                if fusion:
                    print("ReSampling!")
                    lam = 0.1
    #                 pred_z0_avg = pred_z0[random.sample(range(0,16),10)].mean(axis=0) ####original without filtering

                    mc_idx = random.sample(good_indices, min(10, len(good_indices)))
                    fusion_idx = random.sample(mc_idx, min(8, len(mc_idx)))
                    fused_z0_init = pred_z0[mc_idx].mean(axis=0)
                    print(pred_z0.shape, 'z0 shape')
                    print(fused_z0_init.shape, 'passing 1st')
                    fused_x0_init = self.model.decode_first_stage(fused_z0_init[None])
    #                 fused_x0_init = fused_x0_init.clone()
                    print(fused_x0_init.shape, 'passing 2nd')
    #                 fused_x0_init = fused_x0_init.requires_grad_()
                    fused_x0_init = fused_x0_init.detach()
                    fused_x0_init.requires_grad = True

    #                 z = torch.rand_like(fused_x0_init).to(device)
    #                 z.requires_grad = True


                    pred_x0 = self.model.decode_first_stage(pred_z0).detach()

                    loss_fn_lpips = lpips_loss
                    loss_fn_l2 = torch.nn.functional.mse_loss
                    optimizer = torch.optim.AdamW([fused_x0_init], lr=5e-3)

                    for itr in range(125):
                        optimizer.zero_grad()

    #                     output = (z**2).mean()
                        output1 = 0.5 * sum(loss_fn_lpips(img_i, fused_x0_init) for img_i in pred_x0[fusion_idx])
    #                     print(pred_x0[fusion_idx].shape, 'choosing fusion imgs')
                        output2 = 5 * sum(loss_fn_l2(img_i, fused_x0_init) for img_i in pred_x0[fusion_idx])
                        output = output1 + output2
                        output.backward() # Take GD step
                        optimizer.step()
                        cur_loss = output.detach().cpu().numpy()

                        if itr %10 == 0:
                            print(cur_loss, itr)
                    fused_z0 = self.model.get_first_stage_encoding(self.model.encode_first_stage(fused_x0_init))
                    
                    direction = fused_z0 - pred_z0
                    projected_z0 = pred_z0 + lam * direction ####hopefully remain the original shape

                    ####################if purification then do stochastic encode and decode back####################
#                     print(projected_z0.shape)
                    sigma = 20*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev) ####gamma for resample
                    x = self.stochastic_resample(pseudo_x0=projected_z0, x_t=x, a_t=a_t, sigma=sigma)

                else: 
                    print("ReSampling! Pixelavg false, mc false")
                    lam = 0.1
    #                 pred_z0_avg = pred_z0[random.sample(range(0,16),10)].mean(axis=0) ####original without filtering

                    # pred_z0_avg = pred_z0[random.sample(good_indices, min(10, len(good_indices)))].mean(axis=0)
                    pred_z0_avg = pred_z0.mean(axis=0)
                    direction = pred_z0_avg - pred_z0
                    projected_z0 = pred_z0 + lam * direction ####hopefully remain the original shape

                    ####################if purification then do stochastic encode and decode back####################
                    print(projected_z0.shape)
                    sigma = 5*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                    x = self.stochastic_resample(pseudo_x0=projected_z0, x_t=x, a_t=a_t, sigma=sigma)

        
        if vps:
            print("VPS!")
            noise = noise_like(x.shape, device, repeat_noise)
            lam_ = .04
            ###############################compute score#########################
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c, sc = sc, y= y)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                new_c = None
                if type(c) == list:
            ################################lr image as a new condition################################
                    c_in = torch.cat([unconditional_conditioning, c[0]])
                    lr_in = torch.cat([c[1], c[1]])
                    new_c = [c_in, lr_in]
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                    new_c = c_in
            sc_in = None
            if sc is not None:
                sc_in = torch.cat([sc] * 2)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, new_c, sc = sc_in, y=y_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            #############################################################################
            x = x - lam_ * (1 - a_prev).sqrt() * e_t
            x = x + ((lam_ * (2-lam_))*(1-a_prev)).sqrt() * noise * 1
            ###############################update score again#########################
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c, sc = sc, y= y)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                new_c = None
                if type(c) == list:
            ################################lr image as a new condition################################
                    c_in = torch.cat([unconditional_conditioning, c[0]])
                    lr_in = torch.cat([c[1], c[1]])
                    new_c = [c_in, lr_in]
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                    new_c = c_in
            sc_in = None
            if sc is not None:
                sc_in = torch.cat([sc] * 2)
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, new_c, sc = sc_in, y= y_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
            #############################################################################
                
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
#         if resample:
#             pred_x0 = projected_z0
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def encode(self, x0, c, t_enc, sc = None, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
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
                noise_pred = self.model.apply_model(x_next, t, c, sc=sc)
            else:
                assert unconditional_conditioning is not None
                ############compute score###################################
                sc_in = None
                if sc is not None:
                    sc_in = torch.cat([sc] * 2)
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c)), sc = sc_in), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out
    
    

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0[1]).repeat(16,1,1,1)

        device = x0.device
        
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise), noise

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, sc = None):
        """
        
        img, new_cond, ts, index=index, sc = sc, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, vps = vps, resample = resample, pixelavg = pixelavg, blend_param = blend_param, y = y
                                      """

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning, sc = sc, y = None, resample = False, vps = False)
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
            ###convergence criteria####
            
        #####detecting gradient descent converging/overfitting ############
            if itr < 200:
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)                
        ######detecting noise overfitting####################
#             if cur_loss < 0.010**2:   #######change this for noise level###############
#                 break

            # Convergence criteria

        return z_init, init_loss
    
    