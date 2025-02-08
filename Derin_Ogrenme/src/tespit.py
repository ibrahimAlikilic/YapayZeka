import cv2
import numpy as np

# Dosya yolları
weights_path = "../input/yolov4_best.weights"  # YOLO ağırlıkları
config_path = "../darknet/cfg/yolov4.cfg"  # YOLO konfigürasyon dosyası
names_path = "../darknet/data/obj.names"  # Sınıf isimleri dosyası (Özel model için farklı olabilir)
video_path = "../input/test_videosu.mp4"  # İşlenecek video dosyası
output_path = "../output/output.mp4"  # Çıkış videosu

# Sınıf isimlerini yükleme
with open(names_path, "r") as f:
    class_names = f.read().strip().split("\n")

# YOLO modelini yükleme
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # GPU kullanıyorsan: DNN_TARGET_CUDA

# Video yakalama
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video çıkışı için ayarlar
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Katman isimlerini al
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Renkleri tanımla (Her sınıfa rastgele bir renk)
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO için giriş verisi hazırlama
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Tespit edilen nesneleri işleme
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:  # Güven eşiğini 0.3'e düşürdüm
                center_x, center_y, width, height = (
                    detection[0] * frame_width,
                    detection[1] * frame_height,
                    detection[2] * frame_width,
                    detection[3] * frame_height,
                )
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Algılanan nesneleri yazdır
    print("Algılanan sınıf id’leri:", class_ids)
    print("Tespit edilen kutu sayısı:", len(boxes))

    # Aynı nesneye ait fazla kutucukları kaldırma (Non-Maximum Suppression - NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.1)

    print("NMS sonrası kalan kutu sayısı:", len(indices))

    # Çizimleri ekrana ekleme
    if len(indices) > 0:
        for i in indices.flatten():
            if i < len(class_ids):  # Güvenlik kontrolü eklendi
                x, y, w, h = boxes[i]
                label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                print(f"UYARI: class_ids[{i}] geçersiz bir indeks!")

    else:
        print("⚠️ Uyarı: Hiçbir nesne tespit edilmedi!")

    # Sonuçları yazdır
    out.write(frame)

    # Video gösterme (İsteğe bağlı, Google Colab'de kullanamazsın)
    cv2.imshow("YOLOv4 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Temizlik
cap.release()
out.release()
cv2.destroyAllWindows()

'''
import cv2
import numpy as np

# Dosya yolları
weights_path = "../input/yolov4_best.weights"  # YOLO ağırlıkları
config_path = "../darknet/cfg/yolov4.cfg"  # YOLO konfigürasyon dosyası
names_path = "../darknet/data/obj.names"  # Sınıf isimleri dosyası (Özel model için farklı olabilir)
video_path = "../input/test_videosu.mp4"  # İşlenecek video dosyası
output_path = "../output/output.mp4"  # Çıkış videosu

# Sınıf isimlerini yükleme
with open(names_path, "r") as f:
    class_names = f.read().strip().split("\n")

# YOLO modelini yükleme
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # GPU kullanıyorsan: DNN_TARGET_CUDA

# Video yakalama
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video çıkışı için ayarlar
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Katman isimlerini al
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Renkleri tanımla (Her sınıfa rastgele bir renk)
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO için giriş verisi hazırlama
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Tespit edilen nesneleri işleme
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Güven eşiği
                center_x, center_y, width, height = (
                    detection[0] * frame_width,
                    detection[1] * frame_height,
                    detection[2] * frame_width,
                    detection[3] * frame_height,
                )
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aynı nesneye ait fazla kutucukları kaldırma (Non-Maximum Suppression - NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Çizimleri ekrana ekleme
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sonuçları yazdır
    out.write(frame)

    # Video gösterme (İsteğe bağlı, Google Colab'de kullanamazsın)
    cv2.imshow("YOLOv4 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Temizlik
cap.release()
out.release()
cv2.destroyAllWindows()
'''