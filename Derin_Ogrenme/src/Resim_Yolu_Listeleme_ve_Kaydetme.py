import os
from natsort import natsorted # type: ignore # bunun sayesinde dosyaları okuduğumuz sırayla konumunu yazdırıyoruz.

def save_image_paths(directory, output_file):
    # Dosyaları doğal sıralama ile sırala
    files = [f for f in os.listdir(directory) if f.endswith(".jpg")]
    files = natsorted(files)
    
    with open(output_file, 'w') as file:
        for filename in files:
            file.write(os.path.join(directory, filename) + "\n")

# Klasör yolları ve çıktı dosyalarını tanımlayın
train_directory = '../custom_yolo_model/data/traffic_sign/images/resize_TRAIN'
test_directory = '../custom_yolo_model/data/traffic_sign/images/resize_TEST'
train_txt = '../custom_yolo_model/data/traffic_sign/images_paths/training.txt'
test_txt = '../custom_yolo_model/data/traffic_sign/images_paths/testing.txt'

# Eğitim ve test yollarını kaydet
save_image_paths(train_directory, train_txt)
save_image_paths(test_directory, test_txt)

print("Yol kaydetme işlemi tamamlandı.")
