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

parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input', default='/data/qinyi2/projects/attack_sam/data/part', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='/data/qinyi2/projects/attack_sam/adv', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--target', default='/data/qinyi2/projects/attack_sam/adv', type=str, help='directory of crafted adversarial examples')
args = parser.parse_args()
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu' )

#加载模型
model_path = r'/data/qinyi2/projects/attack_sam/segment-anything/sam_vit_b_01ec64.pth'
sam = sam_model_registry["vit_b"](checkpoint=model_path).eval()
sam = sam.to(device)   
#import pdb;pdb.set_trace()
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
def  prepare(imagex,transform):
    imagex = transform.apply_image_torch(imagex) 
    fine = imagex
    return fine.contiguous()

tar_path = r'/data/qinyi2/projects/attack_sam/data/part/sa_4125.jpg'
directory = args.input_dir
adv_output =  args.output_dir
coords = torch.tensor([[200, 150]], dtype=torch.float, device=sam.device).unsqueeze(0) #(x,2)
point_labels = torch.tensor([[1]], device=sam.device)   #(b,n) = (1,1)
tar = cv2.imread(tar_path)
tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
tar_tensor= torch.as_tensor(tar).float()
tar_tensor = tar_tensor.permute(2, 0, 1).unsqueeze(0).to(device) #([1, 3, 1500, 2250])
temp = prepare(tar_tensor,resize_transform)

batched_input_tar = [
            {
         'image': temp, 
        #  'boxes': resize_transform.apply_boxes_torch(image_boxes2, tar.shape[:2]),
         'point_coords': resize_transform.apply_coords_torch(coords, tar.shape[:2]),
         'point_labels': point_labels,
         'original_size':tar.shape[:2]
         }]
batched_output_tar = sam(batched_input_tar, multimask_output=False)
tar_embedding = batched_output_tar[0]['masks'].squeeze(0) #【256，64，64】embedding

alpha = 2
eps = 16
step = 50

for filename in os.listdir(directory):
    name, _ = os.path.splitext(os.path.basename(filename))  #文件名
    if filename.endswith('.jpg') or filename.endswith('.png'): # 检查文件扩展名
    # if filename == "sa_15.jpg":
            image_path = os.path.join(directory, filename)
            image1 = cv2.imread(image_path)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image_sry= torch.as_tensor(image1).float()
            image_sry = image_sry.permute(2, 0, 1).unsqueeze(0).to(device) #bchw 

            # batched_input_cle = [{
            #              'image': prepare(image_sry,resize_transform), 
            #              'point_coords': resize_transform.apply_coords_torch(coords,image1.shape[:2]),
            #              'point_labels': point_labels,
            #              'original_size': image1.shape[:2]
            #              }]
            # batched_output_cle = sam(batched_input_cle, multimask_output=False)
            # cle_embedding = batched_output_cle[0]['masks'].squeeze(0) 

            max_x = image_sry  + eps
            min_x = image_sry  - eps
            with torch.enable_grad():
                imaget = image_sry.detach().clone()
                delta = (torch.rand_like(imaget) * 2 - 1) * eps  # 范围-16-16
                adv = imaget + delta.to(imaget.device)

                for i in range(step):
                    adv_tmp = adv.detach().clone()
                    adv_tmp.requires_grad = True
                    batched_input_adv = [{
                        'image': prepare(adv_tmp, resize_transform),
                        'point_coords': resize_transform.apply_coords_torch(coords, image1.shape[:2]),
                        'point_labels': point_labels,
                        'original_size': image1.shape[:2]
                    }]
                    batched_output_adv = sam(batched_input_adv, multimask_output=False)
                    adv_embedding = batched_output_adv[0]['masks'].squeeze(0)
                    loss =   F.mse_loss(adv_embedding, tar_embedding)  # 有目标，最小
                    # record.append(loss)
                    loss.backward(retain_graph=True)
                    new_grad = adv_tmp.grad
                    adv = adv_tmp - alpha * new_grad.sign()
                    adv = torch.clamp(adv, 0.0, 255.0).detach()
                    adv = torch.max(torch.min(adv, max_x), min_x).detach()
                    import pdb;pdb.set_trace()
                    # adv_tmp.grad.zero_()
                    # torch.cuda.empty_cache()
            # print(record)
            # record.clear()        
            final = adv.squeeze(0) #chw 255
            img = final.float() / 255.0
            img = transforms.ToPILImage()(img.cpu())
            img.save(os.path.join(adv_output, f'latest_{name}.jpg'))
            
    
    
 

















        



        

