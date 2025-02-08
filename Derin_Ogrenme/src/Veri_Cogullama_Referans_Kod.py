import cv2
import albumentations as A
import numpy as np
import os

# Veri çoğullama dönüşümleri
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.RandomSizedCrop(min_max_height=(500, 600), height=608, width=608, p=0.5)
])

# Görüntüleri okuma ve çoğullama işlemi
input_folder = "dataset/original_images"
output_folder = "dataset/augmented_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)
    
    if image is not None:
        augmented = transform(image=image)["image"]
        output_path = os.path.join(output_folder, f"aug_{image_name}")
        cv2.imwrite(output_path, augmented)

print("Veri çoğullama tamamlandı!")
