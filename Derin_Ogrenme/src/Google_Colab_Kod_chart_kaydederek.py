# Hata aldım ve tekrardan yazdım

# Google Drive'a bağlanma
from google.colab import drive
drive.mount('/content/drive')

# Gerekli kütüphanelerin yüklenmesi
import os
import time
import shutil
import threading
import matplotlib.pyplot as plt
import re

# CUDA sürümünü kontrol etme
!nvcc --version
!nvidia-smi

# chart.png dosyasını düzenli olarak kaydetmek için fonksiyon
def save_chart_periodically(source_path, destination_path, interval=300):
    while True:
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        time.sleep(interval)

# chart.png dosyasının kaydedileceği hedef dizin
chart_dest_folder = "/content/drive/MyDrive/Otonom_Egitim_Calisma/sonuclar_grafik/"
os.makedirs(chart_dest_folder, exist_ok=True)
chart_thread = threading.Thread(target=save_chart_periodically, args=("/content/darknet/chart.png", chart_dest_folder))
chart_thread.start()

# Darknet kurulumu
!apt-get update
!rm -rf /content/darknet
!unzip "/content/drive/MyDrive/Otonom_Egitim_Calisma/darknet.zip" -d /content/
%cd /content/darknet


##################################
# Klasör içindeki gerekli düzenlemeleri yap

# Darknet temizleme ve yeniden derleme
!make clean
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!make -j$(nproc)

# Darknet için gerekli bağlantıları kurma
!rm -rf /content/darknet/backup
!ln -s /content/drive/MyDrive/Otonom_Egitim_Calisma/traffic_weights/backup /content/darknet

# Model eğitimi başlatma
train_cmd = "./darknet detector train data/traffic_sign/obj.data yolov4.cfg /content/drive/MyDrive/Otonom_Egitim_Calisma/traffic_weights/backup/yolov4_last.weights -map -dont_show"
try:
    os.system(train_cmd)
except Exception as e:
    print(f"Model eğitimi sırasında hata oluştu: {e}")
















##############################################################################################################################


# Google Colab kodu ile model egitimi yaparken olusan chart.png dosyasini kaydetme ve aynı grafiği kendimiz çizdirerek kaydetme
# Google Drive'a baglanma
from google.colab import drive
drive.mount('/content/drive')

# Gerekli kutuphanelerin ve fonksiyonlarin tanimi
import shutil
import os
import time
import threading

# chart.png dosyasini düzenli olarak kaydetmek için fonksiyon
def save_chart_periodically(source_path, destination_path, interval=300):
    while True:
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        time.sleep(interval)

# chart.png dosyasinin kaydedilecegi hedef dizin
destination_folder = "/content/drive/MyDrive/Otonom_Egitim_Calisma/sonuclar_grafik/"
os.makedirs(destination_folder, exist_ok=True)

# chart.png'yi kaydetmek icin thread'i baslat
chart_thread = threading.Thread(target=save_chart_periodically, args=("/content/darknet/chart.png", destination_folder))
chart_thread.start()

# Darknet icin gerekli kurulumlar
!apt-get update

# Neredeyim
%pwd

!unzip "/content/drive/MyDrive/Otonom_Egitim_Calisma/darknet.zip"
%cd /content/darknet

# Neredeyim
%pwd

!sudo apt install dos2unix
!find . -type f -print0 | xargs -0 dos2unix
!chmod +x /content/darknet
!make

# Darknet icin gerekli baglantilari kurma
!rm /content/darknet/backup -r
!ln -s /content/drive/MyDrive/Otonom_Egitim_Calisma/traffic_weights/backup /content/darknet

# Neredeyim
%pwd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

# Model egitimi baslatma
!./darknet detector train data/traffic_sign/obj.data yolov4.cfg yolov4.conv.137 -map -dont_show
# !./darknet detector train data/traffic_sign/obj.data yolov4.cfg backup/yolov4_last.weights -map -dont_show # elinde hali hazırda belli bir aşamaya kadar eğitilmiş weihts varsa bu


# Model egitimi tamamlandiktan sonra log.txt'den tablo olusturma
import matplotlib.pyplot as plt
import re

# log.txt dosyasindan verileri okumak
log_file = "/content/darknet/log.txt"
iterations = []
loss_values = []

with open(log_file, 'r') as f:
    for line in f:
        match = re.search(r"(\d+): \d+\.\d+, (\d+\.\d+)", line)  # iteration ve loss degerlerini yakalamak icin regex
        if match:
            iterations.append(int(match.group(1)))
            loss_values.append(float(match.group(2)))

# Grafigi olusturmak ve kaydetmek
plt.figure(figsize=(10, 5))
plt.plot(iterations, loss_values, label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()
plt.savefig("/content/drive/MyDrive/Otonom_Egitim_Calisma/charts/training_loss.png")
plt.show()
