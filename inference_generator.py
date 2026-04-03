import os
import cv2
import torch
import random
import pickle
import argparse
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from models.gpstoken import GPSToken
from models.sit_models import SiT_models

from sample import create_transport, Sampler
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpstoken_path", type=str)
    parser.add_argument("--initg_path", type=str)
    parser.add_argument("--max_count", type=int, default=-1)
    parser.add_argument("--data_count", type=int, default=50)
    parser.add_argument("--class_count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=str, default="results/generator/")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--guidance_low", type=int, default=0)
    parser.add_argument("--guidance_high", type=int, default=1000)
    parser.add_argument("--ode", action="store_true")
    # --- ends ---

    args = parser.parse_args()

    # --- set gpu env ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # --- ends ---

    # --- load config ---
    assert args.config.endswith(".yaml")
    args._conf = OmegaConf.load(args.config)
    # --- ends ---

    return args

def main():
    args = parse_args()

    # --- accelerator ---
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    set_seed(args.seed)
    # --- ends ---

    # --- create diffusion ---
    transport = create_transport(
        args._conf.transport.path_type,
        args._conf.transport.prediction,
        args._conf.transport.loss_weight,
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    if args.ode:
        sampler_fn = transport_sampler.sample_ode(num_steps=args.steps)
    else:
        sampler_fn = transport_sampler.sample_sde(num_steps=args.steps)
    # --- ends ---

    # --- load model ---
    model = SiT_models[args._conf.model.name](input_size=(args._conf.model.latent_size,1), num_classes=args._conf.model.num_classes, in_channels=args._conf.model.in_channels, cond_channels=args._conf.model.in_channels-args._conf.gpstoken.gpsconfig.gps_c)
    state_dict = load_file(args.model_path)
    state_dict = {k.replace("ema_model.", ""):v for k,v in state_dict.items() if k.startswith("ema_model.")}
    model.load_state_dict(state_dict)
    model.eval()

    gpstoken = GPSToken(decoderconfig=args._conf.gpstoken.decoderconfig, gpsconfig=args._conf.gpstoken.gpsconfig)

    state_dict = load_file(args.gpstoken_path)
    model_weight = {k: v for k, v in state_dict.items() if k in gpstoken.state_dict()}
    gpstoken.load_state_dict(model_weight, strict=True)
    gpstoken.eval()
    # --- ends ---

    # --- prepare ----
    model, gpstoken = accelerator.prepare(model, gpstoken)
    model_uw, gpstoken_uw = accelerator.unwrap_model(model), accelerator.unwrap_model(gpstoken)
    # --- ends ---

    # --- inference ---
    total_count = args.data_count * args.class_count if args.max_count <= 0 else args.max_count

    # --- for g_init ---
    with open(args.initg_path, "rb") as f:
        ginit_data = pickle.load(f)
    def load_gsp(y):
        return ginit_data[y].pop()
    # --- ends ---

    # ---> one forward func
    rgenerator = torch.Generator(accelerator.device)
    def _forward( ys ):
        b = len(ys)
        rgenerator.manual_seed(ys[0])
        with torch.no_grad():

            # ---> preprocess
            z_cond = torch.randn(b, args._conf.model.in_channels, args._conf.gpstoken.gpsconfig.gps_num, 1, device=accelerator.device, generator=rgenerator)
            zs = torch.cat([z_cond, z_cond], dim=0)
            y_cond = torch.tensor([(i%args.class_count)%args._conf.model.num_classes for i in ys], device=accelerator.device)
            y_uncond = torch.tensor([args._conf.model.num_classes for i in ys], device=accelerator.device)
            y = torch.cat([y_cond, y_uncond], dim=0)

            # ---> load gsp
            gp_c = args._conf.model.in_channels-args._conf.gpstoken.gpsconfig.gps_c
            g_init = [load_gsp((i%args.class_count)%args._conf.model.num_classes) for i in ys]
            g_init = [torch.from_numpy(_gsp).to(accelerator.device).unsqueeze(0)[:,:,:gp_c] for _gsp in g_init]
            g_init = torch.cat(g_init+g_init, dim=0)

            # --->
            model_kwargs = dict(y=y, cond=g_init, cfg_scale=args.cfg_scale, guidance_high=args.guidance_high, guidance_low=args.guidance_low)

            # ---> sample
            samples = sampler_fn(zs, model_uw.forward_with_cfg, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)
            g_init = g_init[:samples.shape[0]]
            samples = samples[:,:,:,0].permute(0,2,1).contiguous()

            # ---> postprocess
            for _i in range(samples.shape[-1]):
                samples[:,:,_i] = samples[:,:,_i]*args._conf.pretrained_gpstoken_stds[_i]+args._conf.pretrained_gpstoken_means[_i]
            samples[:,:,:gp_c] = samples[:,:,:gp_c] + g_init
            samples[:,:,0:2] = torch.clamp(samples[:,:,0:2], 0.00001, 0.99999)
            if gp_c == 5:
                samples[:,:,2] = torch.clamp(samples[:,:,2], -0.99999, 0.99999)

            # ---> decode
            rec = gpstoken_uw.decode(samples)

        for bi in range(b):
            save_path = os.path.join(args.output, f"{ys[bi]}.png")
            cv2.imwrite(save_path, ((rec[bi].clamp(-1,1).cpu().numpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8)[:,:,::-1])


    # distribute the datas 
    all_bi = [list(range(_s,min(_s+args.batch_size, total_count),1)) for _s in range(0, total_count, args.batch_size)]
    current_bi = [all_bi[_s] for _s in range(accelerator.process_index, len(all_bi), accelerator.num_processes)]
    random.shuffle(current_bi)
    pbar = tqdm(len(current_bi), desc=f"GPU {accelerator.process_index}")
    os.makedirs(args.output, exist_ok=True)
    for batchi in current_bi:
        pbar.update(1)

        # skip
        for _i in batchi:
            save_path = os.path.join(args.output, f"{_i}.png")
            if not os.path.exists(save_path):
                break
        else:
            print(f"skip {batchi} ...")
            continue

        _forward(batchi)

    print(f"saving all the samples in {args.output}")

if __name__ == "__main__":
    main()

