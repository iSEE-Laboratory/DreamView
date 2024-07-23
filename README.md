# DreamView
Official PyTorch implementation of ECCV 2024 paper “DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation”. 

![-](assets/2D.png)

This repository includes the training and inference code for text-to-image generation (DreamView-2D), and text-to-3D generation (DreamView-3D).

You can check `DreamView-2D/` and `DreamView-3D/` for the text-to-image and text-to-3D generation codes, respectively.

## News
- 2024/7/22: We release the code and script for training DreamView-2D and text-to-3D generation via DreamView-3D
- 2024/7/3: Our paper is accepted by ECCV 2024, congratulations and many thanks to the co-authors!🥳🎉🎊
- 2024/4/11: We release the inference code of DreamView-2D, and the DreamView checkpoint: [Baidu Drive](https://pan.baidu.com/s/19k9qK7bNNWlChWfe483s9w?pwd=r3ie), [Google Drive](https://drive.google.com/file/d/1MD35zN5niGkV_u77cyLClpVFKjreO1Bf/view?usp=sharing), [ModelScope](https://www.modelscope.cn/models/Drinky/DreamView/summary), [HuggingFace](https://huggingface.co/Drinky/DreamView)
- 2024/4/10: Our arxiv paper can be found [here](https://arxiv.org/abs/2404.06119)

## TODO
- [ ] Release the rendered dataset
- [x] Release the code of DreamView-3D
- [x] Release the training script for training DreamView-2D

## Preliminary
### Environment
- For DreamView-2D: please follow [MVDream](https://github.com/bytedance/MVDream) to prepare the environments
- For DreamView-3D: please follow [threestudio](https://github.com/threestudio-project/threestudio) to prepare the environments

### Checkpoint
- Download the checkpoints at [Baidu Drive](https://pan.baidu.com/s/19k9qK7bNNWlChWfe483s9w?pwd=r3ie), [Google Drive](https://drive.google.com/file/d/1MD35zN5niGkV_u77cyLClpVFKjreO1Bf/view?usp=sharing), [ModelScope](https://www.modelscope.cn/models/Drinky/DreamView/summary), [HuggingFace](https://huggingface.co/Drinky/DreamView)
- Move the checkpoints to `ckpts/`

## Text-to-image Generation
### Inference
Running the below script
```
cd DreamView-2D
python t2i.py --num_samples 4 --fp16
```
and you are expected to obtain the below result:
![-](assets/output-2d.png)
It takes about 10G GPU memory to run the text-to-image generation, and you can modify the `DreamView-2D/t2i.py` to generate your own content.

### Training
```
cd DreamView-2D
bash train.sh
```
Note that we use 4 8*V100 machine to train DreamView-2D by default, and to accelerate convergence, you can consider using [MVDream](https://github.com/bytedance/MVDream) as the initialization parameter.

## Text-to-3D Generation
![-](assets/output-3d.gif)
Running the script below to reproduce the results shown above
```
cd DreamView-3D
bash reproduce.sh
```
Note that the above script may require ~60G GPU memory, so you may run it with an A100 GPU.

## Acknowledgement
- The code of DreamView-2d is heavily based on [MVDream](https://github.com/bytedance/MVDream) and [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).
- The code of DreamView-3d is heavily based on [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio) and [threestudio](https://github.com/threestudio-project/threestudio). 
- We would like to thank the authors for sharing their exciting works.

## Citation
- If you find this repo helpful, please consider citing us:
``` bibtex
@inproceedings{yan2024DreamView,
  author = {Yan, Junkai and Gao, Yipeng and Yang, Qize and Wei, Xihan and Xie, Xuansong and Wu, Ancong and Zheng, Wei-Shi},
  title = {DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation},
  booktitle = {ECCV},
  year = {2024}
}
```
