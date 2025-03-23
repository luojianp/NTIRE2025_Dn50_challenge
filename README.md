# NTIRE2025_Dn50_challenge
This project is about the Team02 test script of NTIRE2025_Dn50_challenge, including the following contents:
1. Model weight file:
   model_zoo/team02_mambairv2.ckpt
3. Model structure file:
   models/mambairv2_arch.py
5. Model test reasoning script:
   python test_NTIRE.py


CUDA_VISIBLE_DEVICES=0 python3 dn/test_NTIRE.py \
    --data_dir test_imgs_dir \
    --mode hybrid_test \
    --model mambair \
    --model_dir models/Mambair_30.367.pth \
    --tile 320 \
    --tile_overlap 32 \
    --out_dir Your_path 

CUDA_VISIBLE_DEVICES=0 python3 dn/test_NTIRE.py \
    --data_dir test_imgs_dir\
    --mode hybrid_test \
    --model promptir \
    --model_dir models/Promptir_Dn50.pth \
    --tile 192 \
    --tile_overlap 32 \
    --out_dir Your_path
