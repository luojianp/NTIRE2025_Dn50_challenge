# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## Preparation：
1. Model download link: wget https://huggingface.co/datasets/luojianping/NTIRE2025_Dn50_Challenge_Team02_Testout/resolve/main/model_zoo.zip
2. Model structure of model file in `./models/team02_promptir_arch.py` and `./models/team02_mambairv2_arch.py`
3. Put the model file in `./model_zoo/team02_Mambair_30.367.pth` and `model_zoo/team02_Promptir_Dn50.pth`

## Test the model：

Execute the script to test from [`run.sh`](./run.sh) 

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

