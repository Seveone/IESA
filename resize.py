from PIL import Image
import os

# 原始目录和目标目录的路径
original_directory = "/data/qinyi2/sorted_images"  # 替换为您的原始目录路径
target_directory = "/data/qinyi2/projects/attack_sam/data/part"  # 替换为您的目标目录路径

# 确保目标目录存在
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 遍历原始目录中的所有JPEG文件
for file in os.listdir(original_directory):
    if file.endswith(".jpg"):
        original_path = os.path.join(original_directory, file)
        target_path = os.path.join(target_directory, file)

        # 打开图片并调整尺寸
        with Image.open(original_path) as img:
            resized_img = img.resize((400, 300), Image.Resampling.LANCZOS)

            # 保存调整后的图片
            resized_img.save(target_path)

print(f"已将图片从 {original_directory} 缩放并保存到 {target_directory}")
