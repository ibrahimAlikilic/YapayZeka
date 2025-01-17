import cv2
import glob
import os

# Yeni boyutlar
target_width, target_height = 416,416

# Görüntü ve etiket klasörlerinin yolunu ayarlayın
image_folder = "C:/Users/ibo_m/OneDrive/Masaüstü/GitHub_VisualStudio/Trafik_isiklari/custom_yolo_model/yolov4/darknet/data/traffic_sign/images/TEST"
label_folder = "C:/Users/ibo_m/OneDrive/Masaüstü/GitHub_VisualStudio/Trafik_isiklari/custom_yolo_model/yolov4/darknet/data/traffic_sign/labels/TEST"
output_image_folder = "C:/Users/ibo_m/OneDrive/Masaüstü/GitHub_VisualStudio/Trafik_isiklari/custom_yolo_model/yolov4/darknet/data/traffic_sign/images/resize_TEST"
output_label_folder = "C:/Users/ibo_m/OneDrive/Masaüstü/GitHub_VisualStudio/Trafik_isiklari/custom_yolo_model/yolov4/darknet/data/traffic_sign/labels/resize_TEST"

# Çıkış klasörlerini oluştur
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

for img_path in glob.glob(f"{image_folder}/*.jpg"): 
    # Görüntüyü oku
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Görüntüyü yeniden boyutlandır
    resized_img = cv2.resize(img, (target_width, target_height))
    
    # Yeni görüntüyü kaydet
    new_img_path = os.path.join(output_image_folder, os.path.basename(img_path))
    cv2.imwrite(new_img_path, resized_img)
    
    # Etiket dosyasını oku
    label_path = os.path.join(label_folder, os.path.basename(img_path).replace('.jpg', '.txt'))
    new_label_path = os.path.join(output_label_folder, os.path.basename(label_path))
    
    if not os.path.exists(label_path):
        print("Etiket dosyası bulunamadı.")
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        # Her bir etiket satırını parçalara ayır
        class_id, x_norm, y_norm, width_norm, height_norm = map(float, line.strip().split())
        
        # Normalize değerlerden piksel değerlerine geçiş
        x = x_norm * w
        y = y_norm * h
        width = width_norm * w
        height = height_norm * h
        
        # Yeniden boyutlandırma oranına göre değerleri güncelle
        x = x * target_width / w
        y = y * target_height / h
        width = width * target_width / w
        height = height * target_height / h
        
        # Güncel değerleri normalize et
        x_norm_new = x / target_width
        y_norm_new = y / target_height
        width_norm_new = width / target_width
        height_norm_new = height / target_height
        
        # Yeni etiket satırını ekle
        new_lines.append(f"{int(class_id)} {x_norm_new:.6f} {y_norm_new:.6f} {width_norm_new:.6f} {height_norm_new:.6f}/n")
    
    # Yeni etiket dosyasını kaydet
    with open(new_label_path, 'w') as f:
        f.writelines(new_lines)

print("İşlem tamamlandı.")



"""
1. Normalize Değerlerden Piksel Değerlerine Geçiş
Kod:

python
Kodu kopyala
x = x_norm * w
y = y_norm * h
width = width_norm * w
height = height_norm * h
Açıklama:
Normalize edilmiş değerler (x_norm, y_norm, width_norm, height_norm) 0 ile 1 arasında bir aralıkta tutulur. Bu değerler orijinal görüntü boyutlarına göre hesaplanmıştır.
Normalize değerleri gerçek piksel değerlerine dönüştürmek için, orijinal görüntünün genişliği (w) ve yüksekliği (h) kullanılır.
Merkez koordinatları (x, y) normalize edildiğinde, her bir değer orijinal görüntü genişliği veya yüksekliğine oranlanarak belirlenmiştir.
Genişlik ve yükseklik (width, height) de aynı mantıkla normalize edilmiştir.


2. Yeniden Boyutlandırma Oranına Göre Güncelleme
Kod:

python
Kodu kopyala
x = x * target_width / w
y = y * target_height / h
width = width * target_width / w
height = height * target_height / h
Açıklama:
Görüntü, hedef genişlik ve yüksekliğe (target_width, target_height) yeniden boyutlandırılmıştır. Bu işlem sırasında, nesne koordinatlarının ve boyutlarının da yeni boyutlara uygun şekilde yeniden ölçeklenmesi gerekir.
Bu nedenle, eski piksel değerleri yeni boyutlara göre oranlanır.
Merkez koordinatları (x, y) yeni genişlik ve yükseklik ile orantılı şekilde yeniden hesaplanır.
Nesne boyutları (width, height) da aynı mantıkla yeni boyutlara uyarlanır.


3. Güncellenen Değerlerin Normalize Edilmesi
Kod:

python
Kodu kopyala
x_norm_new = x / target_width
y_norm_new = y / target_height
width_norm_new = width / target_width
height_norm_new = height / target_height
Açıklama:
Yeni boyutlarda hesaplanan piksel değerlerini tekrar normalize etmek gerekir. Bu, YOLO gibi model formatlarının normalize edilmiş değerlerle çalışmasından kaynaklanır.
Normalize işlemi, yeni genişlik ve yükseklik (target_width, target_height) kullanılarak yapılır. Böylece değerler tekrar 0 ile 1 arasına getirilir.
"""