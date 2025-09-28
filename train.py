import os
import sys
import torch 
import argparse
import numpy as np
from loguru import logger
from ema_pytorch import EMA
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from models.gpstoken import GPSToken
from losses.create_loss import init_perceptualer
from datasets.create_dataset import init_dataset

# ---- load taming-transformers ----
taming_dir = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-1])
taming_dir = os.path.join(taming_dir, "taming-transformers")
sys.path.insert(1, taming_dir)
print(sys.path)
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
logger.info(f"using taming from {taming_dir}")
# ---- ends ----

def get_newest_folder(path, key=""):
    folders = os.listdir(path)
    folders = [f for f in folders if os.path.isdir(os.path.join(path, f))]
    folders = [f for f in folders if key in f]
    folders = [os.path.join(path, f) for f in folders]
    folders = sorted(folders, key=lambda x: os.path.getmtime(x), reverse=True)
    return folders[0] if len(folders) > 0 else None

def parse_args():

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--project", type=str, default="project")
    parser.add_argument("--config", type=str, default="/home/notebook/code/personal/S9049747/projects/GSVAE/taming-transformers/data/imagenet_val.txt")
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
    parser.add_argument("--output", type=str, default="experiments")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs.")
    # --- ends ---

    args = parser.parse_args()

    # --- set gpu env ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # --- ends ---

    # ----- proc the output_dir -----
    args.output = os.path.join(args.output, f"{args.project}")
    args.model_dir = os.path.join(args.output, "models")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    # ----- ends -----

    # --- load config ---
    assert args.config.endswith(".yaml")
    args._conf = OmegaConf.load(args.config)
    # --- ends ---

    # --- set logger ---
    logger.add(os.path.join(args.output, "train.log"), rotation="10 MB")
    # --- ends ---

    return args

def main():
    args = parse_args()

    # --- accelerator ---
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    set_seed(args.seed)
    # --- ends ---

    # --- load train dataset ---
    _config = {
        "target": args._conf.data.target,
        "params": {
            "gps_num": args._conf.model.gpsconfig.gps_num,
            "adaptive_weight": args._conf.model.gpsconfig.gps_adaptive_w,
            "paths_or_file": args._conf.data.params.paths_or_file,
            "size": args._conf.data.params.size,
            "hflip": True,
            }
    }
    dataset = init_dataset(_config)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args._conf.train.batch_size, shuffle=True, num_workers=args._conf.data.num_workers)
    loader = accelerator.prepare(loader)
    loader_iter = iter(loader)
    logger.info(f"Dataset loaded: {len(dataset)}")
    # --- ends ---

    # --- load model ---
    model = GPSToken(decoderconfig=args._conf.model.decoderconfig, gpsconfig=args._conf.model.gpsconfig)
    model.train()
    ema_model = EMA(model)

    discriminator = VQLPIPSWithDiscriminator(**args._conf.loss.discriminator)
    discriminator.train()

    perceptualer = init_perceptualer(args._conf.loss.perceptual)
    perceptualer.eval()
    # --- ends ---

    # --- optimizer ---
    opt_ae = torch.optim.Adam(
            [p for n,p in model.named_parameters()],
            lr=args._conf.train.lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(discriminator.discriminator.parameters(),
                lr=args._conf.train.lr, betas=(0.5, 0.9))
    # --- ends ---

    # --- prepare ----
    model, ema_model, discriminator, perceptualer, opt_ae, opt_disc = accelerator.prepare(model, ema_model, discriminator, perceptualer, opt_ae, opt_disc)
    # --- ends ---

    # --- resume ---
    resume_dir = get_newest_folder(args.model_dir, key="checkpionts.")
    if args.auto_resume and resume_dir is not None:
        accelerator.load_state(resume_dir)
        start_step = int(resume_dir.split(".")[-1])
        logger.info(f"Resume from {resume_dir}, step: {start_step}")
    else:
        start_step = 0
    # --- ends ---

    # --- set grad accumulation ---
    if hasattr(args._conf.train, "grad_accumulation"):
        grad_accumulation = args._conf.train.grad_accumulation
    else:
        grad_accumulation = 1
    logger.info(f"Set the grad accumulation to {grad_accumulation}")
    # --- ends ---

    # --- training ---
    step = start_step
    _step_woac = -1 # step without accumulation
    while(step < args._conf.train.iterations):

        # --- load data ---
        try:
            data = loader_iter.__next__()
        except StopIteration:
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args._conf.train.batch_size, shuffle=True, num_workers=args._conf.data.num_workers)
            loader = accelerator.prepare(loader)
            loader_iter = iter(loader)
            data = loader_iter.__next__()
        # --- ends ---


        # 0 for autoencode and 1 for discriminator
        optimizer_idx = step%2 if step > args._conf.loss.discriminator.disc_start else 0
        if _step_woac % grad_accumulation == 0: # clear the grad
            if optimizer_idx == 0: opt_ae.zero_grad()
            else: opt_disc.zero_grad()

        # --- one-step ---
        model_input = data["image"].permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        init_gpscodes = data["gpscodes"].to(model_input.device).float()
        regions = data["regions"].to(model_input.device)

        xrec = model(model_input, init_gpscodes=init_gpscodes, regions=regions)

        # discriminator
        if hasattr(model, 'module'):
            loss, loss_log = discriminator(0.0*torch.mean(model_input), model_input, xrec, optimizer_idx, step, last_layer=model.module.get_last_layer_weight(), split="train")
        else:
            loss, loss_log = discriminator(0.0*torch.mean(model_input), model_input, xrec, optimizer_idx, step, last_layer=model.get_last_layer_weight(), split="train")

        # perceptual loss
        perceptual_loss = perceptualer(xrec, model_input)
        loss = loss + perceptual_loss
        loss_log["perceptual_loss"] = perceptual_loss
        # --- ends --- 

        # loss.backward()
        accelerator.backward(loss/grad_accumulation)

        # optimizer step
        if _step_woac % grad_accumulation == grad_accumulation-1:
            if optimizer_idx == 0: 
                opt_ae.step()
                if hasattr(ema_model, "module"):
                    ema_model.module.update()
                else:
                    ema_model.update()
            else: 
                opt_disc.step()

        if ((_step_woac%grad_accumulation==grad_accumulation-1 and step%args._conf.train.log_interval in (0,1)) 
            or _step_woac<grad_accumulation) and accelerator.is_main_process:
            _log = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_log.items()])
            logger.info(f"Step {step}, optimizer_idx:{optimizer_idx}, {_log}")

        if step % args._conf.train.save_interval == 0 and step!=0:
            save_dir = os.path.join(args.model_dir, f"checkpionts.{step:08d}")
            accelerator.save_state(save_dir)

        _step_woac += 1
        if _step_woac%grad_accumulation == grad_accumulation-1:
            step += 1

    # save the last model
    save_dir = os.path.join(args.model_dir, f"checkpionts.{step:08d}")
    accelerator.save_state(save_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if Accelerator().is_main_process:
            import traceback; traceback.print_exc()
            print(e)
            import pdb; pdb.post_mortem()
