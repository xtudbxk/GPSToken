import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from safetensors.torch import load_file

from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from models.gpstoken import GPSToken
from datasets.create_dataset import init_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=str, default="gpstoken_results")
    parser.add_argument("--gpus", type=str, default="0")
    # --- ends ---

    args = parser.parse_args()

    # --- set gpu env ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # --- ends ---

    # ----- proc the output_dir -----
    os.makedirs(args.output, exist_ok=True)
    args.imgs_dir = os.path.join(args.output, "imgs")
    os.makedirs(args.imgs_dir, exist_ok=True)
    args.gt_dir = os.path.join(args.output, "gt")
    os.makedirs(args.gt_dir, exist_ok=True)
    # ----- ends -----

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

    # --- load dataset ---
    _config = {
        "target": "AdaptiveGPSDataset",
        "params": {
            "gps_num": args._conf.model.gpsconfig.gps_num,
            "adaptive_weight": args._conf.model.gpsconfig.gps_adaptive_w,
            "paths_or_file": args.data_path,
            "size": args.data_size,
            "hflip": False,
            }
    }
    dataset = init_dataset(_config)
    logger.info(f"Dataset loaded: {len(dataset)}")
    # --- ends ---

    # --- load model ---
    model = GPSToken(decoderconfig=args._conf.model.decoderconfig, gpsconfig=args._conf.model.gpsconfig)
    model.eval()

    state_dict = load_file(args.model_path)
    model_weight = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(model_weight, strict=True)
    # --- ends ---

    # --- prepare ----
    model = accelerator.prepare(model)
    model_uw = accelerator.unwrap_model(model)
    # --- ends ---

    # --- inference ---
    allids = list(range(accelerator.process_index, len(dataset), accelerator.num_processes))
    pbar = tqdm(len(allids), desc=f"GPU {accelerator.process_index}")
    for ids in allids:
        pbar.update(1)

        with torch.no_grad():
            data = dataset[ids]

            # --- forward ---
            img = torch.from_numpy(data["image"]).to(accelerator.device)[None].permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
            init_gpscodes = torch.from_numpy(data["gpscodes"]).to(img.device)[None].float()
            regions = torch.from_numpy(data["regions"]).to(img.device)[None]

            # ---> fold higher resolution into 256x256 patch
            _np = args.data_size // 256
            _b, _c, _h, _w = img.shape
            _img = img.reshape(_b,_c,_np,256,_np,256).permute(0,2,4,1,3,5).reshape(_b*_np*_np,_c,256,256)
            _, _gs, _gc = init_gpscodes.shape
            init_gpscodes = init_gpscodes.reshape(-1,_gs//(_np*_np), _gc)
            _, _, _rc = regions.shape
            regions = regions.reshape(-1,_gs//(_np*_np), _rc)

            # ---> encode
            gpstokens = model_uw.encode(_img, init_gpscodes=init_gpscodes, regions=regions) # [_b*_np*_np,gsn,gsc]

            # ---> rendering & decode
            if int(args._conf.model.gpsconfig.gps_num) == 256:
                # gpstoken-L256 dose not need special processing for higher resolution due to high reconstruction performance
                xrec = model_uw.decode(gpstokens, _np=1)
                _b2, _c2, _h2, _w2 = xrec.shape
                xrec = xrec.reshape(_b2//(_np**2),_np,_np,_c2,_h2,_w2)
                xrec = xrec.permute(0,3,1,4,2,5)
            else:
                xrec = model_uw.decode(gpstokens, _np=_np)
            xrec = xrec.reshape(_b,-1,args.data_size,args.data_size)

            # save the image
            file_path = data['path']
            filedir, filename = os.path.split(file_path)
            filename = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(args.imgs_dir, filename)
            cv2.imwrite(save_path, ((xrec[0].clamp(-1,1).cpu().numpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8)[:,:,::-1])
            save_path = os.path.join(args.gt_dir, filename)
            cv2.imwrite(save_path, ((img[0].clamp(-1,1).cpu().numpy().transpose(1,2,0)+1.0)*127.5).astype(np.uint8)[:,:,::-1])

            # 
    logger.info(f"write images to {args.imgs_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if Accelerator().is_main_process:
            import traceback; traceback.print_exc()
            print(e)
            import pdb; pdb.post_mortem()
