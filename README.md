# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## Preparation：
1. Download the model file, Model download link: wget https://huggingface.co/datasets/luojianping/NTIRE2025_Dn50_Challenge_Team02_Testout/resolve/main/model_zoo.zip
2. Model structure of model file in `./models/team02_promptir_arch.py` and `./models/team02_mambairv2_arch.py`
3. Put the model file in `./model_zoo/team02_Mambair_30.367.pth` and `model_zoo/team02_Promptir_Dn50.pth`

## Test the model：

Execute the script to test from [`run.sh`](./run.sh) 

# Step 1: Execute inference of the mambair model. Line 48 of the code is: model_type = 'mambair'. The default is: model_type = 'mambair'
CUDA_VISIBLE_DEVICES=0 python3 test_demo.py \
--data_dir ./NTIRE2025_Challenge/input \
--model_id 2 \
--save_dir ./NTIRE2025_Challenge/results

# Step 2: Execute inference of the promptir model. Line 48 of the code is: model_type = 'promptir'
CUDA_VISIBLE_DEVICES=0 python3 test_demo.py \
--data_dir ./NTIRE2025_Challenge/input \
--model_id 2 \
--save_dir ./NTIRE2025_Challenge/results

# Step 3: Take the results of the first and second steps 0.5 times each, and then add them together to get the final effect.
