import os
import logging
import torch
import argparse
import json
import glob
import os.path as osp
import numpy as np
import cv2
import torch.nn.functional as F
import math
import time


from models.team02_promptir_arch import PromptIR
from models.team02_mambairv2_arch import MambaIRv2
from PIL import Image
import datetime

from pprint import pprint
from utils.model_summary import get_model_activation, get_model_flops


"""
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


"""


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# ===============================
# logger
# logger_name = None = 'base' ???
# ===============================
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


# ----------
# PSNR
# ----------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ----------
# SSIM
# ----------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def crop_image(image, s=8):
    h, w, c = image.shape
    image = image[:h - h % s, :w - w % s, :]
    return image


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img
    


class ImageDenoise():
    def __init__(self, args):
        model_dir = args.model_dir
        self.img_multiple_of = 8
        self.tile = args.tile
        self.tile_overlap = args.tile_overlap
        # self.sigma = 50
        is_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if is_cuda else 'cpu')

        if args.model == 'promptir':
           self.model_G = PromptIR().to(self.device)
        elif args.model == 'mambair':
            self.model_G = MambaIRv2().to(self.device)

        if is_cuda:  # load model on gpu
            lm = torch.load(model_dir)
        else:
            lm = torch.load(model_dir, map_location=torch.device('cpu'))
        self.model_G.load_state_dict(lm.state_dict(), strict=True)
        # self.model_G.load_state_dict(lm['params'], strict=True)    # for restormer
        self.model_G.eval()

        """
        使用 Self-Ensemble 进行预测
        :param x: 输入图像 (B, C, H, W)
        :return: Self-Ensemble 的预测结果
        """
        # 8种变换：原始、水平翻转、垂直翻转、旋转90°、旋转180°、旋转270°及其组合
        self.transforms = [
            lambda v: v,  # 原始
            lambda v: v.flip(-1),  # 水平翻转
            lambda v: v.flip(-2),  # 垂直翻转
            lambda v: v.transpose(-1, -2),  # 旋转90°
            lambda v: v.flip(-1).transpose(-1, -2),  # 旋转90° + 水平翻转
            lambda v: v.flip(-2).transpose(-1, -2),  # 旋转90° + 垂直翻转
            lambda v: v.flip(-1).flip(-2),  # 旋转180°
            lambda v: v.flip(-1).flip(-2).transpose(-1, -2),  # 旋转270°
        ]

    
    def inference(self, input_img):
        def tile_inference(input_img, tile_size):
            # print("Do tiling in ", tile_size)
            # test the image tile by tile
            b, c, h, w = input_img.shape
            tile = min(tile_size, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"

            stride = tile - self.tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_img)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = self.model_G(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            pred = E.div_(W)
            return pred

        # 处理 tile 参数
        if self.tile is None:
            tile_list = None
        elif isinstance(self.tile, int):
            tile_list = [self.tile]
        elif isinstance(self.tile, str):
            # 将字符串解析为整数列表
            tile_list = [int(t.strip()) for t in self.tile.split(",")]
        else:
            tile_list = self.tile  # 假设已经是列表
        
        if self.tile is None:
            pred = self.model_G(input_img)
        else:
            preds = [tile_inference(input_img, tile_size) for tile_size in tile_list]
            pred = sum(preds) / len(preds)
        
        return pred
    

    def _inverse_transform(self, transform):
        """
        获取变换的逆操作
        :param transform: 变换函数
        :return: 逆变换函数
        """
        if transform == self.transforms[0]:  # 原始
            return lambda v: v
        elif transform == self.transforms[1]:  # 水平翻转
            return lambda v: v.flip(-1)
        elif transform == self.transforms[2]:  # 垂直翻转
            return lambda v: v.flip(-2)
        elif transform == self.transforms[3]:  # 旋转90°
            return lambda v: v.transpose(-1, -2)
        elif transform == self.transforms[4]:  # 旋转90° + 水平翻转
            return lambda v: v.transpose(-1, -2).flip(-1)
        elif transform == self.transforms[5]:  # 旋转90° + 垂直翻转
            return lambda v: v.transpose(-1, -2).flip(-2)
        elif transform == self.transforms[6]:  # 旋转180°
            return lambda v: v.flip(-1).flip(-2)
        elif transform == self.transforms[7]:  # 旋转270°
            return lambda v: v.transpose(-1, -2).flip(-1).flip(-2)
        else:
            raise ValueError("Unknown transform")

    
    def __call__(self, input_img):
            

        with torch.no_grad():    
        
            input_img = input_img / 255
            input_ = torch.from_numpy(input_img.astype(np.float32).transpose((2, 0, 1))).unsqueeze(0).to(self.device)
            # # Pad the input if not_multiple_of 8
            height, width = input_.shape[2], input_.shape[3]
     
            # 存储所有变换后的预测结果
            outputs = []

            for transform in self.transforms:
                # 对输入图像进行变换
                transformed_input = transform(input_)
                # 使用模型进行预测
                transformed_output = self.inference(transformed_input)
                # 对预测结果进行反变换
                inverse_transform = self._inverse_transform(transform)
                outputs.append(inverse_transform(transformed_output))
            

            pred = torch.stack(outputs, dim=0).mean(dim=0)
                   
            pred = torch.clamp(pred, 0, 1)

            # Unpad the output
            pred = pred[:,:,:height,:width]
            pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])

            # pred = add_sharpening(pred)

            pred = np.round(np.clip(pred * 255, 0, 255)).astype(np.uint8)
            return pred


def select_dataset(data_dir, mode, args):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_test_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
        # [f"DIV2K_test_LR/{i:04}.png" for i in range(901, 1001)]
    elif mode == "valid":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_valid_noise50/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_valid_HR/{i:04}.png")
            ) for i in range(801, 901)
        ]
    elif mode == "hybrid_test":
        path = [
            (
                p,
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    return path


def run(args, denoise, out_dir, logger, mode):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode, args)
    os.makedirs(out_dir, exist_ok=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_noisy, img_hr) in enumerate(data_path):
        print(img_noisy, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # --------------------------------
        # (1) img_noisy
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_noisy))
        input_img = np.array(Image.open(img_noisy))
        input_img = crop_image(input_img)

        # --------------------------------
        # (2) img_dn
        # --------------------------------
        start.record()
        img_dn = denoise(input_img)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds

        if mode != "hybrid_test":
        
            # --------------------------------
            # (3) img_hr
            # --------------------------------
            img_hr = np.array(Image.open(img_hr))
            img_hr = crop_image(img_hr)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            # print(img_dn.shape, img_hr.shape)
            psnr = calculate_psnr(img_dn, img_hr, border=border)
            results[f"{mode}_psnr"].append(psnr)

            if args.ssim:
                ssim = calculate_ssim(img_dn, img_hr, border=border)
                results[f"{mode}_ssim"].append(ssim)
                logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
            else:
                logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        else:
            print("Doing {:s}.".format(img_name + ext))

        # save denoised image
        img_dir = os.path.join(out_dir, img_name+ext)
        # cv2.imwrite(img_dir, img_dn)
        Image.fromarray(img_dn).save(img_dir)

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    if mode != "hybrid_test":
        results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    if mode != "hybrid_test":
        logger.info("Average PSNR {:.2f} dB".format(results[f"{mode}_ave_psnr"]))                       # PSNR
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memery", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))

    return results


def main(args):
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = osp.join(osp.dirname(args.model_dir), "results_psnr")
    os.makedirs(out_dir, exist_ok=True)
    logger_info("NTIRE2025-Dn50", log_path=osp.join(out_dir,"NTIRE2025-Dn50.log"))
    logger = logging.getLogger("NTIRE2025-Dn50")

    # --------------------------------
    # basic settings
    # --------------------------------
    # torch.cuda.current_device()
    # torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    json_dir = os.path.join(out_dir, "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)
    
    denoise = ImageDenoise(args)


    results[args.model] = run(args, denoise, out_dir, logger, args.mode)


    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(denoise.model_G, input_dim)
    activations = activations/10**6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(denoise.model_G, input_dim, False)
    flops = flops/10**9
    logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), denoise.model_G.parameters()))
    num_parameters = num_parameters/10**6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
    results[args.model].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

    with open(json_dir, "w") as f:
        json.dump(results, f)

    if args.mode != "valid":
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        # print(v.keys())
        if args.mode != "valid":
            val_psnr = f"{v['hybrid_test_ave_psnr']:2.2f}"
            val_time = f"{v['hybrid_test_ave_runtime']:3.2f}"
            mem = f"{v['hybrid_test_memory']:2.2f}"
        else:
            val_psnr = f"{v['valid_ave_psnr']:2.2f}"
            val_time = f"{v['valid_ave_runtime']:3.2f}"
            mem = f"{v['valid_memory']:2.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.mode != "valid":
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(osp.join(out_dir, 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-Dn50")
    parser.add_argument("--data_dir", default="./NTIRE2025_Challenge/input", type=str)
    parser.add_argument("--mode", default="valid", type=str)
    parser.add_argument("--model", default="promtir", type=str)
    parser.add_argument("--model_dir", default="exp/Promptir_gate/Model_last.pth", type=str)
    parser.add_argument("--tile", default=None, type=str)
    parser.add_argument("--tile_overlap", default=32, type=int)
    parser.add_argument("--ssim", default=False, help="Calculate SSIM")
    parser.add_argument('--out_dir', help="Output directory", default="", type=str)

    args = parser.parse_args()
    pprint(args)

    main(args)
