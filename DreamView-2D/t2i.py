# Copyright (c) Alibaba, Inc. and its affiliates.
# @author:  Drinky Yan
# @contact: yanjk3@mail2.sysu.edu.cn
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch

from ldm.camera_utils import get_camera
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, sampler, injection, step=20, scale=7.5, batch_size=8, ddim_eta=0.,
        dtype=torch.float32, device="cuda", camera=None, num_frames=1, margin=-0.025):
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        assert injection in [-1, 0, 1, 2]

        c = model.get_learned_conditioning(prompt).to(device)
        global_conds, view_conds = c[0], c[1:]

        uc = model.get_learned_conditioning(["low quality"] * batch_size).to(device)

        if injection == -1:
            c_ = {"context": global_conds.repeat(batch_size, 1, 1), 'global_context': None}
            uc_ = {"context": uc, 'global_context': None}
        if injection == 0:
            c_ = {"context": view_conds, 'global_context': None}
            uc_ = {"context": uc, 'global_context': None}
        if injection == 1:
            global_conds = global_conds.repeat(batch_size, 1, 1)
            c_ = {"context": torch.cat([view_conds, global_conds], dim=1), 'global_context': None}
            uc_ = {"context": torch.cat([uc, uc], dim=1), 'global_context': None}
        if injection == 2:
            global_conds = global_conds.repeat(batch_size, 1, 1)
            c_ = {"context": view_conds, 'global_context': global_conds}
            uc_ = {"context": uc, 'global_context': uc}

        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
            c_["margin"] = uc_["margin"] = margin

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                         batch_size=batch_size, shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc_,
                                         eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='configs/dreamview-32gpus.yaml')
    parser.add_argument("--ckpt_path", type=str, default='../ckpts/dreamview.pth', help="path to checkpoint")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=0)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--injection", type=int, default=2)
    parser.add_argument("--margin", type=float, default=-0.025)
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    print("load t2i model ... ")
    assert args.ckpt_path is not None, "ckpt_path must be specified!"
    config = OmegaConf.load(args.config_path)
    model = instantiate_from_config(config.model)
    del model.model_ema
    msg = model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'), False)
    print('Loading the full model', msg)
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(args.num_frames, elevation=args.camera_elev,
                            azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
    else:
        camera = None

    global_text = "A bulldog wearing a tie and carrying a backpack on the back" + args.suffix

    view_1_text = "A bulldog wearing a tie" + args.suffix
    view_2_text = "A bulldog carrying a backpack on the back" + args.suffix
    view_3_text = "A bulldog carrying a backpack on the back" + args.suffix
    view_4_text = "A bulldog carrying a backpack on the back" + args.suffix

    t = [global_text, view_1_text, view_2_text, view_3_text, view_4_text]

    set_seed(args.seed)
    images = []
    for j in range(args.num_samples):
        img = t2i(model, args.size, t, sampler, injection=args.injection, step=50, scale=7.5, batch_size=batch_size,
                  ddim_eta=0.0, dtype=dtype, device=device, camera=camera, num_frames=args.num_frames,
                  margin=args.margin)
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f'output-{int(time.time())}.png')
