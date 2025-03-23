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

