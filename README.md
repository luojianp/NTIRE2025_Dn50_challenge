# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

This is an example of adding noise and simple baseline model.

## How to add noise to images?
`
python add_noise.py
`

## How to test the baseline model?


Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
CUDA_VISIBLE_DEVICES=0 python3 test_NTIRE.py \
    --data_dir test_imgs_dir \
    --mode hybrid_test \
    --model mambair \
    --model_dir model_zoo/team02_Mambair_30.367.pth \
    --tile 320 \
    --tile_overlap 32 \
    --out_dir Your_path 

CUDA_VISIBLE_DEVICES=0 python3 test_NTIRE.py \
    --data_dir test_imgs_dir\
    --mode hybrid_test \
    --model promptir \
    --model_dir model_zoo/team02_Promptir_Dn50.pth \
    --tile 192 \
    --tile_overlap 32 \
    --out_dir Your_path


    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
   
## How to add your model to this baseline?
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1XVa8LIaAURYpPvMf7i-_Yqlzh-JsboG0hvcnp-oI9rs/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/team02_promptir_arch.py` and `./models/team02_mambairv2_arch.py`
   - Please add ****_arch.py** file in the folder `./models`. 
3. Put the pretrained model in `./model_zoo/team02_Mambair_30.367.pth` and `model_zoo/team02_Promptir_Dn50.pth`
4. The command to download code:
   - `git clone https://github.com/luojianp/NTIRE2025_Dn50_challenge.git`
   - We will do the following steps to add your code and model checkpoint to the repository.
This repository shows how to add noise to synthesize the noisy image. It also shows how you can save an image.
