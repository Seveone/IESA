import matplotlib.pyplot as plt
import argparse
import torch
import random
import shutil
import cv2
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import functional as F
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch


parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_adv_dir', default='/data/qinyi2/projects/attack_sam/adv/tar_adv/cos', type=str, help='directory of clean examples')
parser.add_argument('--input_cle_dir', default='/data/qinyi2/projects/attack_sam/data/part', type=str, help='directory of clean examples')
parser.add_argument('--output_adv_dir', default='/data/qinyi2/projects/attack_sam/seg_out/adv_out/tar_out_pgd/cos', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--output_cle_dir', default='/data/qinyi2/projects/attack_sam/seg_out/clean_out', type=str, help='directory of crafted adversarial examples')
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu' )

#加载模型
model_path = r'/data/qinyi2/projects/attack_sam/segment-anything/sam_vit_b_01ec64.pth'
sam = sam_model_registry["vit_b"](checkpoint=model_path)
sam = sam.to(device)   
#import pdb;pdb.set_trace()

resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
def prepare(imagex,transform):
    imagex = transform.apply_image_torch(imagex)  #裁剪到指定大小
    fine = imagex
    return fine.contiguous()

tar_path = r'/data/qinyi2/projects/attack_sam/data/sa_47.jpg'
adv_directory = args.input_adv_dir
cle_directory = args.input_cle_dir
adv_output =  args.output_adv_dir
cle_output = args.output_cle_dir
#此攻击方式目标与干净共用一个box
# image_boxes1 = torch.tensor([
#     [0, 218, 440, 534]
# ], device=sam.device)  #dog
coords = torch.tensor([[170,120]], dtype=torch.float, device=sam.device).unsqueeze(0) #(x,2)
point_labels = torch.tensor([[1]], device=sam.device)   #(b,n) = (1,1)
# tar = cv2.imread(tar_path)
# tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
# tar_tensor= torch.as_tensor(tar).float()
# tar_tensor = tar_tensor.permute(2, 0, 1).unsqueeze(0).to(device) #1
# batched_input_tar = [
#             {
#          'image': prepare(tar_tensor,resize_transform), 
#         #  'boxes': resize_transform.apply_boxes_torch(image_boxes2, tar.shape[:2]),
#          'point_coords': resize_transform.apply_coords_torch(coords, tar.shape[:2]),
#          'point_labels': point_labels,
#          'original_size':tar.shape[:2]
#          }]

# batched_output_tar = sam(batched_input_tar, multimask_output=False)
# # import pdb;pdb.set_trace()
# tar_out = batched_output_tar[0]['masks'].squeeze(0)
# tar_out = transforms.ToPILImage()(tar_out.cpu())
# tar_out.save(os.path.join("/data/qinyi2/projects/attack_sam/seg_out", 'B_TAR_OUT.jpg'))
 #【256，64，64】e
# import pdb;pdb.set_trace()
# for filename in os.listdir(adv_directory):
#     name, _ = os.path.splitext(os.path.basename(filename))  #文件名
#     if filename.endswith('.jpg') or filename.endswith('.png'): # 检查文件扩展名
#             image_path = os.path.join(adv_directory, filename)
#             image1 = cv2.imread(image_path)
#             image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#             image_sry= torch.as_tensor(image1).float()
#             image_sry = image_sry.permute(2, 0, 1).unsqueeze(0).to(device) #bchw ; 13hw
#             batched_input_adv = [{
#                          'image': prepare(image_sry,resize_transform), 
#                          'point_coords': resize_transform.apply_coords_torch(coords, image1.shape[:2]),
#                          'point_labels': point_labels,
#                          'original_size': image1.shape[:2]
#                          }]
#             batched_output_adv = sam(batched_input_adv, multimask_output=False)
#             adv= batched_output_adv[0]['masks'].squeeze(0) 
#             adv= transforms.ToPILImage()(adv.cpu())
#             adv.save(os.path.join(adv_output, f'{name}.jpg'))
                    

for filename in os.listdir(cle_directory):
    name, _ = os.path.splitext(os.path.basename(filename))  #文件名
    if filename.endswith('.jpg') or filename.endswith('.png'): # 检查文件扩展名
            image_path = os.path.join(cle_directory, filename)
            image1 = cv2.imread(image_path)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image_sry= torch.as_tensor(image1).float()
            image_sry = image_sry.permute(2, 0, 1).unsqueeze(0).to(device) #bchw ; 13hw
            batched_input_cle = [{
                         'image': prepare(image_sry,resize_transform), 
                         'point_coords': resize_transform.apply_coords_torch(coords, image1.shape[:2]),
                         'point_labels': point_labels,
                         'original_size': image1.shape[:2]
                         }]
            batched_output_cle = sam(batched_input_cle, multimask_output=False)
            cle= batched_output_cle[0]['masks'].squeeze(0) 
            cle = transforms.ToPILImage()(cle.cpu())
            cle.save(os.path.join(cle_output, f'{name}.jpg'))

