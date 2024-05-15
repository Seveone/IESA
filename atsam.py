import matplotlib.pyplot as plt
import argparse
import torch
import timm
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
from torch.nn.functional import cosine_similarity
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_dir', default='/data/qinyi2/projects/attack_sam/data/part', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='/data/qinyi2/projects/attack_sam/adv/non_adv/attacksam', type=str, help='directory of crafted adversarial examples')
args = parser.parse_args()
directory = args.input_dir
adv_output =  args.output_dir
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu' )
 #加载模型 'samvit_base_patch16', 'samvit_base_patch16_224', 'samvit_huge_patch16', 'samvit_large_patch16'
# model = timm.create_model(
#     'samvit_base_patch16.sa1b',
#     pretrained=True,
#     num_classes=0,  
# )
# sam_encoder  = model.to(device)


model_path = r'/data/qinyi2/projects/attack_sam/segment-anything/sam_vit_b_01ec64.pth'
sam = sam_model_registry["vit_b"](checkpoint=model_path).to(device)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

def  prepare(imagex,transform):
    imagex = transform.apply_image_torch(imagex) 
    fine = imagex
    return fine.contiguous()

coords = torch.tensor([[170, 120]], dtype=torch.float, device=sam.device).unsqueeze(0) #(x,2)
point_labels = torch.tensor([[1]], device=sam.device) 

alpha = 2
eps = 16
step = 50

for filename in os.listdir(directory):
    name, _ = os.path.splitext(os.path.basename(filename))  
    if filename.endswith('.jpg') or filename.endswith('.png'): 
    # if filename == "sa_15.jpg":
            image_path = os.path.join(directory, filename)
            image1 = cv2.imread(image_path)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) #255
            image_sry= torch.as_tensor(image1).float()
            image_sry = image_sry.permute(2, 0, 1).unsqueeze(0).to(device) 
            #bchw 
            max_x = image_sry  + eps  #300 400
            min_x = image_sry  - eps

            with torch.enable_grad():
                imaget = image_sry.detach().clone()  #400
                delta = (torch.rand_like(imaget) * 2 - 1) * eps  # 范围-16/255-16/255 400
                adv_tensor = imaget + delta.to(imaget.device)  

                # import pdb;pdb.set_trace()# #tensor 400
                for i in range(step):
                    adv_tmp = adv_tensor.detach().clone()
                    adv_tmp.requires_grad_(True) #400
                    batched_input_adv = [{
                        'image': prepare(adv_tmp, resize_transform),
                        'point_coords': resize_transform.apply_coords_torch(coords, image1.shape[:2]),
                        'point_labels': point_labels,
                        'original_size': image1.shape[:2]
                    }]
                    batched_output_adv = sam(batched_input_adv, multimask_output=False)
                    adv_mask= batched_output_adv[0]['masks'].squeeze(0)
                    tensor2 = torch.full((1, 300, 400), -10).float().to(device)
                    #256*64*64
                    # import pdb;pdb.set_trace()
                    loss = F.mse_loss(adv_mask, tensor2)
                    # loss = cosine_similarity(adv_embedding, sry_embedding).mean() #越大越更好
                    print(loss) # 有目标，最小
                    # import pdb;pdb.set_trace()
                    sam.zero_grad()
                    loss.backward()
                    # import pdb;pdb.set_trace()
                    adv_tensor = adv_tmp - alpha * adv_tmp.grad.sign()  
                    adv_tensor = torch.clamp(adv_tensor, 0.0, 255.0).detach()
                    adv_tensor = torch.max(torch.min(adv_tensor, max_x), min_x).detach()
                          
            final =adv_tensor.squeeze(0) #chw 0-1
            img = final.float() /255
            img = transforms.ToPILImage()(img.cpu())
            img.save(os.path.join(adv_output, f'{name}.jpg'))