import cv2
import mediapipe as mp

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Torso/upper body landmark indices and connections
TORSO_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]
TORSO_CONNECTIONS = [
    (11, 13), (13, 15),  # Left shoulder-elbow-wrist
    (12, 14), (14, 16),  # Right shoulder-elbow-wrist
    (11, 12),            # Shoulders
    (23, 24),            # Hips
    (11, 23), (12, 24),  # Shoulder-hip (body frame)
]

# Setup video capture
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert the color
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with both models
        image.flags.writeable = False
        hand_results = hands.process(image)
        pose_results = pose.process(image)
        image.flags.writeable = True

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw only torso/upper body pose landmarks and connections
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Draw torso landmarks
            for idx in TORSO_LANDMARKS:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)

            # Draw torso connections
            for start, end in TORSO_CONNECTIONS:
                lm_start = landmarks[start]
                lm_end = landmarks[end]
                x1, y1 = int(lm_start.x * w), int(lm_start.y * h)
                x2, y2 = int(lm_end.x * w), int(lm_end.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Show the image
        cv2.imshow('Upper Body & Finger Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
            break

cap.release()
cv2.destroyAllWindows()