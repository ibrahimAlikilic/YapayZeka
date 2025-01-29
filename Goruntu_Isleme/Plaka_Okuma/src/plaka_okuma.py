import cv2
import numpy as np
import pytesseract
import imutils

# Kamera akışını başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı temsil eder

if not cap.isOpened():
    print("Kamera açılamadı!")
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

    # döngüde ilk önce countours lara yaklaşıcaz sonra kapalı şekilleri arayacağız
    # yaklaşmak yani düzgün hale getiricez
    screen = None
    for c in cnts:
        # deneysel formul kullanacağız
        epsilon=0.018*cv2.arcLength(c,True) # arvLenght demek yay uzunluğu demek , True ile boşluk var mı yok muya bakıyoz burada 
        approx=cv2.approxPolyDP(c,epsilon,True) # approx yani yaklaşım
        if len(approx)==4: # 4 yakın köşe varsa yani 4 değer saklıysadikdörtgendir dedik
            screen = approx
            break

    if screen is not None:
        # Mask oluştur ve plaka bölgesini ayır
        mask=np.zeros(gray.shape,np.uint8) # mask ile her yeri siyah yaptım
        # şimdi plakanın olduğu yeri beyaz yapacağız
        new_frame=cv2.drawContours(mask,[screen],0,(255,255,255),-1)
        # şimdi plaka bölgesindeki yazıyı counter alanına yapıştıracağım
        new_frame=cv2.bitwise_and(frame,frame,mask=mask)

        # Plaka bölgesini kırp
        x, y = np.where(mask == 255)
        (topX, topY) = (np.min(x), np.min(y))
        (bottomX, bottomY) = (np.max(x), np.max(y))
        cropped = gray[topX:bottomX+1, topY:bottomY+1] # crooped=kırpılmış , 1 fazla alıyoruz ki son değer de gelsin

        # OCR ile plaka okumaya çalış
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

    # Görselleri göster
    cv2.imshow("frame", frame)
    cv2.imshow("edge", edge)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
