import cv2
import mediapipe as mp
import numpy as np
import time

# Initialisiere MediaPipe Hand-Modul
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Öffne die Webcam
cap = cv2.VideoCapture(0)

# Überprüfe, ob die Webcam erfolgreich geöffnet wurde
if not cap.isOpened():
    print("Fehler beim Zugriff auf die Webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Bildes.")
            break
        
        # Konvertiere das Bild in RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Erkenne Hände im Bild
        results = hands.process(rgb_frame)

        # Überprüfe, ob Hände erkannt wurden
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Zeichne die Hand-Landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Hier kannst du die Geste erkennen, z.B. durch spezifische Landmark-Punkte
                # Beispiel für eine einfache Geste (z.B. Daumen hoch):
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Erkennung der Geste (Beispiel: Daumen hoch)
                if thumb_tip.y < index_tip.y:  # Daumen ist höher als Zeigefinger
                    cv2.putText(frame, "Geste: Daumen hoch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Speichern des Bildes
                    timestamp = int(time.time())
                    cv2.imwrite(f"gesture_thumb_up_{timestamp}.jpg", frame)
                    print(f"Geste erkannt! Bild gespeichert als gesture_thumb_up_{timestamp}.jpg")

        # Zeige den Frame an
        cv2.imshow('Hand Gesture Recognition', frame)

        # Breche die Schleife bei Tastendruck 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
