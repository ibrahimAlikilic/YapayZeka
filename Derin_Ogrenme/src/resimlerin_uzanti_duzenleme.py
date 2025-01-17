import os
from PIL import Image

# Klasör yolunu belirtin
folder_path = '../archive/traffic_sign/images/TEST'

# Klasördeki tüm dosyaları döngüye al
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Sadece "png" veya "jpg" formatındaki dosyaları işle
    if filename.lower().endswith(('.png', '.jpg')):
        # Görüntüyü aç
        with Image.open(file_path) as img:
            
            # Yeni dosya adını ".jpg" uzantısıyla oluştur
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Görüntüyü JPEG formatında kaydet
            img.convert('RGB').save(new_file_path, 'JPEG')
        
        # Eğer dosya "png" ise eski dosyayı sil
        if filename.lower().endswith('.png'):
            os.remove(file_path)

print("Dönüşüm tamamlandı.")
