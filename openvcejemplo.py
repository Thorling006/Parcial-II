import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def detectar_letra(contorno):
    hull = cv2.convexHull(contorno, returnPoints=False)
    if hull is None:
        return ""
    defects = cv2.convexityDefects(contorno, hull)
    if defects is None:
        return "A"
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > 10000:
            count_defects += 1
    if count_defects == 0:
        return "Hola"
    elif count_defects == 1:
        return "Me llamo Juan"
    elif count_defects == 2:
        return "Un gusto conocerte"
    elif count_defects == 3:
        return "estoy aprendiendo"
    else:
        return "Gracias"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Definir región de interés (por ejemplo, cuadro en la derecha)
    roi = frame[100:400, 300:600]  # Ajusta según tu cámara

    # Convertir a HSV y crear máscara para color de piel
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Suavizar máscara
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Contornos sobre la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    letra = ""
    if contours:
        c = max(contours, key=cv2.contourArea)
        letra = detectar_letra(c)
        cv2.drawContours(roi, [c], -1, (0, 255, 0), 2)

    cv2.putText(frame, f"Letra: {letra}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar ROI y frame completo
    cv2.rectangle(frame, (300, 100), (600, 400), (255, 0, 0), 2)
    cv2.imshow("Señas simplificadas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Señas simplificadas", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
