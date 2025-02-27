import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=3)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=5)
# Video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame detected")
        break

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand Detection
    hand_result = hand.process(rgb_img)
    if hand_result.multi_hand_landmarks:
        for landmark in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{len(hand_result.multi_hand_landmarks)} hands", 
                    (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No hands detected", (50, 100), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    # Face Detection
    face_result = face_mesh.process(rgb_img)
    if face_result.multi_face_landmarks:
        for landmark in face_result.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, landmark, mp_face_mesh.FACEMESH_TESSELATION)
        cv2.putText(frame, f"{len(face_result.multi_face_landmarks)} faces", 
                    (50, 130), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (50, 130), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.imshow("Face & Hand Detection", frame)
    if cv2.waitKey(20) == ord("k"):
        break
cap.release()
cv2.destroyAllWindows()
