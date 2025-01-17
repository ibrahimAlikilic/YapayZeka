import os
import glob

# Görüntü ve etiket klasörlerini ayarlayın
image_folder = "../archive/traffic_sign/images/TRAIN"
label_folder = "../archive/traffic_sign/labels/TRAIN"

# Eksik etiket dosyalarını tutmak için bir liste
missing_labels = []

# Tüm görüntü dosyalarını döngüye al
for img_path in glob.glob(f"{image_folder}/*.jpg"):
    # İlgili etiket dosyasının adını oluştur
    label_path = os.path.join(label_folder, os.path.basename(img_path).replace('.jpg', '.txt'))
    
    # Eğer etiket dosyası yoksa listeye ekle
    if not os.path.exists(label_path):
        missing_labels.append(label_path)

# Eksik etiket dosyalarını yazdır
print("Eksik etiket dosyaları:", missing_labels)

# boş txt oluşturma
for label_path in missing_labels:
    # Boş bir txt dosyası oluştur
    with open(label_path, 'w') as f:
        pass