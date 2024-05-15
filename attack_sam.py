import matplotlib.pyplot as plt
import torch
import random
import shutil
import cv2
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import functional as F
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu' )

# model_path = r'/data/qinyi/myattack/checkpoints/sam_vit_h_4b8939.pth'
model_path = r'/data/qinyi/attack_sam/sam_vit_b_01ec64.pth'
image_path = r'/data/qinyi/attack_sam/dog.jpg'
tar_path = r'/data/qinyi/attack_sam/bool.jpg'
# tar_path = r'/data/qinyi/attack_sam/Padded Image.png'
seg_output = r"/data/qinyi/attack_sam/seg_out"
image1 = cv2.imread(image_path)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
tar = cv2.imread(tar_path)
tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
# tar = np.transpose(tar, (2,0,1))
# tar = torch.from_numpy(tar)
# tar = tar[:1,:]
# tar = (tar/255).squeeze(0).to(device)  # (534,800)(0-1)
sam = sam_model_registry["vit_b"](checkpoint=model_path)
sam = sam.to(device)  # tensor_shape = (1,3,534,800) 
#import pdb;pdb.set_trace()
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
image_sry = image1.copy() #534 800 3 
image_sry = resize_transform.apply_image(image_sry)
image_sry = torch.as_tensor(image_sry).float()
image_sry = image_sry.permute(2, 0, 1).to(device)
delta = torch.zeros(image_sry.shape,requires_grad = True,device = device)
 #[684, 1024, 3]定义扰动尺寸（）为图片处理之后的大小 ->[3,684,1024]
image_boxes1 = torch.tensor([
    [0, 218, 440, 534]
], device=sam.device)
image_boxes2 = torch.tensor([
    [475, 132, 658, 253]
], device=sam.device)

step = 50
Neg_th = -10
alpha = 2
epsilon = 8 
def prepare_image(imagex,delta,transform):
    imagex = transform.apply_image(imagex)  #裁剪到指定大小
    image_tem = torch.as_tensor(imagex).float().to(device) #hwc[684, 1024, 3]
    image_tem = image_tem.permute(2, 0, 1) #(3,h,w)
    fine = image_tem + delta
    return fine.contiguous() # (3,h,w).contiguous() 
def prepare(imagex,transform):
    imagex = transform.apply_image(imagex)  #裁剪到指定大小
    image_tem = torch.as_tensor(imagex).float().to(device) #hwc[684, 1024, 3]
    image_tem = image_tem.permute(2, 0, 1) #(3,h,w)
    fine = image_tem
    return fine.contiguous()
batched_input_tar = [
            {
         'image': prepare(tar, resize_transform),  #image1为np(1-255)图像，只优化delta,
         'boxes': resize_transform.apply_boxes_torch(image_boxes2, tar.shape[:2]),
         'original_size': tar.shape[:2]
         }]

# loss = prepare_image(image1, delta, resize_transform).sum()
batched_output_tar = sam(batched_input_tar, multimask_output=False)
tar = batched_output_tar[0]['masks'].squeeze(0) # (1,534,800) (bool)
# tar = tar*255
# tar = tar.float()
# save_image(tar,'/data/qinyi/attack_sam/seg_out/tar.png')
# import pdb;pdb.set_trace()
tar  = (tar*1.0).squeeze(0).to(device)

for j in range(step):
        delta.requires_grad = True
        batched_input = [
            {
         'image': prepare_image(image1, delta, resize_transform),  #image1为np(1-255)图像，只优化delta,
         'boxes': resize_transform.apply_boxes_torch(image_boxes1, image1.shape[:2]),
         'original_size': image1.shape[:2]
         }]
        batched_output = sam(batched_input, multimask_output=False)
        a = batched_output[0]['masks'].squeeze(0) # (1,534,800) (bool)
        masks  = (a*1.0).squeeze(0)   # (1,534,800) (0-1)
        loss = torch.norm(masks - tar)
        # print(delta.grad)
        # print(delta.is_leaf)
        print(loss)  
        loss.backward(retain_graph=True)
        # import pdb;pdb.set_trace()
        print(delta.grad)
        
        grad = delta.grad
        delta= torch.clamp(delta - alpha * torch.sign(grad), min=-epsilon, max=epsilon) #最小化mse
        # 
        if delta.grad is not None:
            delta.grad.zero_()  # 清零梯度
        delta = delta.detach()
        

fina = delta.data 
batched_input2 = [
            {
         'image': prepare_image(image1, fina, resize_transform),  
         'boxes': resize_transform.apply_boxes_torch(image_boxes1, image1.shape[:2]),
         'original_size': image1.shape[:2]
         }]
batched_output2 = sam(batched_input, multimask_output=False)
b = batched_output2[0]['masks'].squeeze(0)
b = b*255
b = b.float()
save_image(b,'/data/qinyi/attack_sam/seg_out/tar_advout.png')




