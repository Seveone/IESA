import torch
from torchvision import transforms
from PIL import Image
import os
adv_dir ='/data/qinyi2/projects/attack_sam/adv/tar_adv/cos'
adv_mask_dir = '/data/qinyi2/projects/attack_sam/seg_out/adv_out/tar_out_pgd/cos'
tar_mask_dir = r'/data/qinyi2/projects/attack_sam/seg_out/clean_out/sa_4125.jpg'
cle_mask_dir = '/data/qinyi2/projects/attack_sam/seg_out/clean_out'
new_dir = '/data/qinyi2/projects/attack_sam/adv_part_pgd_cos'
files = os.listdir(adv_mask_dir)
os.makedirs(new_dir, exist_ok=True)
transform = transforms.ToTensor()  
tar_mask = Image.open(tar_mask_dir)     
tar = transform(tar_mask)           

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou
           
ious = []                
for filename in os.listdir(adv_mask_dir):
    name, _ = os.path.splitext(os.path.basename(filename))  #文件名
    if filename.endswith('.jpg') or filename.endswith('.png'):
         adv_mask_path = os.path.join(adv_mask_dir, filename)
         adv_mask = Image.open(adv_mask_path)
         adv = transform(adv_mask)
        #  b = transform(b_mask)
         iou = calculate_iou(adv, tar)
         ious.append(iou)

miou = torch.mean(torch.tensor(ious))
print(miou)