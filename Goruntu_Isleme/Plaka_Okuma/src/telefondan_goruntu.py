# ip wabcam uygulamasını kullanarak telefondan görüntü alacağım

import cv2
import numpy as np
import pytesseract
import imutils

# IP Webcam URL'si (telefon uygulamasındaki IP ve portu buraya yaz)
url = "http://192.168.114.2:8080/video"

# IP kameradan görüntü almak için VideoCapture başlat
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Kamera açılamadı! IP adresini ve bağlantıyı kontrol edin.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Kenarları yumuşat ve keskinleştir
    filtered = cv2.bilateralFilter(gray, 6, 250, 250)

    # Kenar tespiti
    edge = cv2.Canny(filtered, 30, 200)

    # Konturların bulunması
    contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(contours)
    cnts = sorted(cnts1, key=cv2.contourArea, reverse=True)[:10]

    screen = None
    for c in cnts:
        epsilon = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            screen = approx
            break

    if screen is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_frame = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)
        new_frame = cv2.bitwise_and(frame, frame, mask=mask)

        x, y = np.where(mask == 255)
        (topX, topY) = (np.min(x), np.min(y))
        (bottomX, bottomY) = (np.max(x), np.max(y))
        cropped = gray[topX:bottomX+1, topY:bottomY+1]

        try:
            text = pytesseract.image_to_string(cropped, lang="eng").strip()
            if text:
                print("Plaka:", text)
                cv2.rectangle(frame, (topY, topX), (bottomY+1, bottomX+1), (0, 255, 0), 2)
                cv2.putText(frame, text, (topY, topX-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print("Plaka okunamadı!")
        except Exception as e:
            print("OCR hatası:", e)
    frame_goster=cv2.resize(frame,(640,520))
    edge_goster=cv2.resize(edge,(640,520))
    cv2.imshow("frame_goster", frame_goster)
    cv2.imshow("edge", edge_goster)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
