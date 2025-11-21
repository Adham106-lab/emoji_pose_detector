"""
Real-time Emoji Display based on Camera Pose and Facial Expression Detection
Using MediaPipe for Face Mesh and Hand Tracking
Author: ADham Omar
Adham106-lab
insta:adhamomar1112
Date: November 2025

Requirements:
pip install mediapipe opencv-python numpy

PNG files needed in 'emojis' folder:
- happy.png, sleepy.png, surprised.png, shocked.png
- look_left.png, look_right.png, look_up.png, look_down.png
- peace.png, thumbs_up.png, fist.png, palm.png
- pointing.png, ok_sign.png, rock.png, wave.png

Optional: meme_background.jpg (default meme background image)
"""

import mediapipe as mp
import cv2
import numpy as np
import os
from pathlib import Path

# Initialize MediaPipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Emoji configurations with detection logic
class EmojiDetector:
    def __init__(self, emoji_folder='emojis'):
        self.emoji_folder = emoji_folder
        self.emoji_cache = {}
        self.current_pose = None
        self.meme_background = None

        # Define all pose patterns with their PNG filenames
        self.poses = [
            {'name': 'Happy', 'file': 'happy.png', 'type': 'face'},
            {'name': 'Sleepy', 'file': 'sleepy.png', 'type': 'face'},
            {'name': 'Surprised', 'file': 'surprised.png', 'type': 'face'},
            {'name': 'Shocked', 'file': 'shocked.png', 'type': 'face'},
            {'name': 'Look Left', 'file': 'look_left.png', 'type': 'face'},
            {'name': 'Look Right', 'file': 'look_right.png', 'type': 'face'},
            {'name': 'Look Up', 'file': 'look_up.png', 'type': 'face'},
            {'name': 'Look Down', 'file': 'look_down.png', 'type': 'face'},
            {'name': 'Peace Sign', 'file': 'peace.png', 'type': 'hand'},
            {'name': 'Thumbs Up', 'file': 'thumbs_up.png', 'type': 'hand'},
            {'name': 'Fist', 'file': 'fist.png', 'type': 'hand'},
            {'name': 'Open Palm', 'file': 'palm.png', 'type': 'hand'},
            {'name': 'Pointing', 'file': 'pointing.png', 'type': 'hand'},
            {'name': 'OK Sign', 'file': 'ok_sign.png', 'type': 'hand'},
            {'name': 'Rock On', 'file': 'rock.png', 'type': 'hand'},
            {'name': 'Wave', 'file': 'wave.png', 'type': 'hand'},
        ]

        self.load_emojis()
        self.load_meme_background()

    def load_meme_background(self):
        """Load meme background image or create a default one"""
        meme_path = os.path.join(self.emoji_folder, 'meme_background.jpg')

        if os.path.exists(meme_path):
            self.meme_background = cv2.imread(meme_path)
            print(f"✓ Loaded meme_background.jpg")
        else:
            # Create a default colorful gradient background
            self.meme_background = self.create_default_meme_background()
            print("ℹ Using default meme background (add meme_background.jpg for custom)")

    def create_default_meme_background(self, width=1280, height=720):
        """Create a default colorful meme-style background"""
        # Create gradient background
        background = np.zeros((height, width, 3), dtype=np.uint8)

        # Create colorful gradient (blue to purple to pink)
        for y in range(height):
            ratio = y / height
            # RGB gradient
            r = int(100 + 155 * ratio)  # 100 to 255
            g = int(50 + 100 * (1 - ratio))  # 150 to 50
            b = int(200 - 100 * ratio)  # 200 to 100
            background[y, :] = [b, g, r]

        # Add some noise/texture for meme effect
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return background

    def load_emojis(self):
        """Load all emoji images with transparency support"""
        if not os.path.exists(self.emoji_folder):
            os.makedirs(self.emoji_folder)
            print(f"Created '{self.emoji_folder}' folder. Please add PNG files.")
            return

        for pose in self.poses:
            filepath = os.path.join(self.emoji_folder, pose['file'])
            if os.path.exists(filepath):
                # Load with alpha channel
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.emoji_cache[pose['file']] = img
                    print(f"✓ Loaded {pose['file']}")
                else:
                    print(f"✗ Failed to load {pose['file']}")
            else:
                print(f"✗ Missing {pose['file']}")

    def detect_face_pose(self, face_landmarks, frame_shape):
        """Detect facial expression and head pose"""
        if not face_landmarks:
            return None

        h, w = frame_shape[:2]
        landmarks = face_landmarks.landmark

        # Key landmark indices for face mesh
        LEFT_EYE_UPPER = 159
        LEFT_EYE_LOWER = 145
        RIGHT_EYE_UPPER = 386
        RIGHT_EYE_LOWER = 374

        MOUTH_TOP = 13
        MOUTH_BOTTOM = 14
        MOUTH_LEFT = 61
        MOUTH_RIGHT = 291

        LEFT_EYEBROW = 70
        RIGHT_EYEBROW = 300

        NOSE_TIP = 1
        CHIN = 152
        FOREHEAD = 10

        # Calculate eye aspect ratios
        left_eye_height = abs(landmarks[LEFT_EYE_UPPER].y - landmarks[LEFT_EYE_LOWER].y)
        right_eye_height = abs(landmarks[RIGHT_EYE_UPPER].y - landmarks[RIGHT_EYE_LOWER].y)
        avg_eye_height = (left_eye_height + right_eye_height) / 2

        # Calculate mouth aspect ratio
        mouth_height = abs(landmarks[MOUTH_TOP].y - landmarks[MOUTH_BOTTOM].y)
        mouth_width = abs(landmarks[MOUTH_LEFT].x - landmarks[MOUTH_RIGHT].x)

        # Calculate head rotation (simplified)
        nose_x = landmarks[NOSE_TIP].x

        # Detect poses
        # 1. Eyes closed (Sleepy)
        if avg_eye_height < 0.01:
            return 'sleepy.png'

        # 2. Mouth open (Surprised)
        if mouth_height > 0.04:
            return 'surprised.png'

        # 3. Smiling (Happy) - mouth wider than normal
        if mouth_width > 0.15 and mouth_height < 0.03:
            return 'happy.png'

        # 4. Eyebrows raised (Shocked)
        eyebrow_y = (landmarks[LEFT_EYEBROW].y + landmarks[RIGHT_EYEBROW].y) / 2
        if eyebrow_y < landmarks[FOREHEAD].y + 0.02:
            return 'shocked.png'

        # 5. Head direction
        if nose_x < 0.35:
            return 'look_left.png'
        elif nose_x > 0.65:
            return 'look_right.png'

        # Vertical head pose
        nose_y = landmarks[NOSE_TIP].y
        if nose_y < 0.35:
            return 'look_up.png'
        elif nose_y > 0.65:
            return 'look_down.png'

        return None

    def detect_hand_gesture(self, hand_landmarks):
        """Detect hand gestures based on finger positions"""
        if not hand_landmarks:
            return None

        landmarks = hand_landmarks.landmark

        # Finger tip and base indices
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        THUMB_BASE = 2
        INDEX_BASE = 5
        MIDDLE_BASE = 9
        RING_BASE = 13
        PINKY_BASE = 17

        WRIST = 0

        def is_finger_extended(tip_idx, base_idx):
            """Check if finger is extended"""
            return landmarks[tip_idx].y < landmarks[base_idx].y

        def finger_distance(idx1, idx2):
            """Calculate distance between two landmarks"""
            return np.sqrt(
                (landmarks[idx1].x - landmarks[idx2].x)**2 +
                (landmarks[idx1].y - landmarks[idx2].y)**2
            )

        # Count extended fingers
        index_up = is_finger_extended(INDEX_TIP, INDEX_BASE)
        middle_up = is_finger_extended(MIDDLE_TIP, MIDDLE_BASE)
        ring_up = is_finger_extended(RING_TIP, RING_BASE)
        pinky_up = is_finger_extended(PINKY_TIP, PINKY_BASE)
        thumb_up = landmarks[THUMB_TIP].x < landmarks[THUMB_BASE].x  # Simplified thumb check

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # 1. Peace sign (index and middle up, others down)
        if index_up and middle_up and not ring_up and not pinky_up:
            return 'peace.png'

        # 2. Pointing (only index up)
        if index_up and not middle_up and not ring_up and not pinky_up:
            return 'pointing.png'

        # 3. Open palm (all fingers extended)
        if fingers_up == 4 and thumb_up:
            return 'palm.png'

        # 4. Fist (no fingers extended)
        if fingers_up == 0:
            return 'fist.png'

        # 5. Thumbs up (only thumb extended)
        if thumb_up and fingers_up == 0:
            return 'thumbs_up.png'

        # 6. OK sign (thumb and index touching)
        thumb_index_dist = finger_distance(THUMB_TIP, INDEX_TIP)
        if thumb_index_dist < 0.05 and middle_up and ring_up and pinky_up:
            return 'ok_sign.png'

        # 7. Rock on (index and pinky up)
        if index_up and pinky_up and not middle_up and not ring_up:
            return 'rock.png'

        # 8. Wave (all fingers extended, moving)
        if fingers_up >= 3:
            return 'wave.png'

        return None

    def create_meme_output(self, emoji_file, pose_name):
        """Create meme-style output with emoji on background"""
        # Resize background to match output size
        meme = cv2.resize(self.meme_background.copy(), (1280, 720))

        if emoji_file not in self.emoji_cache:
            # No emoji detected, show default message
            cv2.putText(meme, "NO POSE DETECTED", (320, 360),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4)
            return meme

        emoji = self.emoji_cache[emoji_file]

        # Make emoji large (center of screen)
        emoji_size = 400
        emoji_resized = cv2.resize(emoji, (emoji_size, emoji_size))

        # Center position
        center_x = 1280 // 2
        center_y = 720 // 2

        x1 = center_x - emoji_size // 2
        y1 = center_y - emoji_size // 2
        x2 = x1 + emoji_size
        y2 = y1 + emoji_size

        # Overlay emoji with transparency
        if emoji_resized.shape[2] == 4:
            alpha = emoji_resized[:, :, 3] / 255.0
            for c in range(3):
                meme[y1:y2, x1:x2, c] = (
                    alpha * emoji_resized[:, :, c] +
                    (1 - alpha) * meme[y1:y2, x1:x2, c]
                )
        else:
            meme[y1:y2, x1:x2] = emoji_resized[:, :, :3]

        # Add pose name text at bottom
        text = pose_name.upper()
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        thickness = 4

        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (1280 - text_width) // 2
        text_y = 650

        # Draw text with outline (meme style)
        # Black outline
        cv2.putText(meme, text, (text_x-2, text_y-2), font, font_scale, (0, 0, 0), thickness+2)
        cv2.putText(meme, text, (text_x+2, text_y+2), font, font_scale, (0, 0, 0), thickness+2)
        # White text
        cv2.putText(meme, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return meme

    def overlay_emoji(self, frame, emoji_file, position, size=150):
        """Overlay emoji with transparency on frame"""
        if emoji_file not in self.emoji_cache:
            return frame

        emoji = self.emoji_cache[emoji_file]

        # Resize emoji
        emoji_resized = cv2.resize(emoji, (size, size))

        x, y = position
        x = max(0, min(x - size // 2, frame.shape[1] - size))
        y = max(0, min(y - size // 2, frame.shape[0] - size))

        # Handle alpha channel for transparency
        if emoji_resized.shape[2] == 4:
            alpha = emoji_resized[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+size, x:x+size, c] = (
                    alpha * emoji_resized[:, :, c] +
                    (1 - alpha) * frame[y:y+size, x:x+size, c]
                )
        else:
            frame[y:y+size, x:x+size] = emoji_resized[:, :, :3]

        return frame


def show_startup_menu():
    """Display startup menu and get user choice"""
    print("\n" + "="*60)
    print(" "*15 + "EMOJI POSE DETECTOR")
    print("="*60)
    print("\n📸 TRACKING MODE SELECTION:\n")
    print("  1. Face Tracking Only")
    print("  2. Hand Tracking Only")
    print("  3. Both Face and Hand Tracking")
    print("\n" + "="*60)

    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("❌ Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main function to run real-time emoji detection"""

    # Show startup menu
    choice = show_startup_menu()

    # Set tracking modes based on choice
    if choice == '1':
        show_face = True
        show_hands = False
        mode_name = "Face Tracking Only"
    elif choice == '2':
        show_face = False
        show_hands = True
        mode_name = "Hand Tracking Only"
    else:  # choice == '3'
        show_face = True
        show_hands = True
        mode_name = "Both Face and Hand"

    print("\n" + "="*60)
    print(f"✓ Starting with: {mode_name}")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'h' - Toggle hand tracking")
    print("  'f' - Toggle face tracking")
    print("\nWindows:")
    print("  Camera Feed - Live video with tracking landmarks")
    print("  Emoji Output - Meme-style static background with emoji")
    print("\nPose Detection:")
    print("  16 total poses (8 face + 8 hand gestures)")
    print("="*60 + "\n")

    # Initialize detector
    detector = EmojiDetector()

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize MediaPipe
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, \
    mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detected_emoji = None
            emoji_position = None
            pose_name = "No pose detected"
            pose_type = None

            # Process face
            if show_face:
                face_results = face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Draw face mesh
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                        # Detect face pose
                        emoji_file = detector.detect_face_pose(face_landmarks, frame.shape)
                        if emoji_file:
                            detected_emoji = emoji_file
                            pose_type = 'face'
                            # Position emoji above face
                            nose = face_landmarks.landmark[1]
                            emoji_position = (int(nose.x * w), int(nose.y * h) - 100)
                            pose_name = next((p['name'] for p in detector.poses if p['file'] == emoji_file), "Unknown")

            # Process hands
            if show_hands:
                hand_results = hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                        # Detect hand gesture
                        emoji_file = detector.detect_hand_gesture(hand_landmarks)
                        if emoji_file and not detected_emoji:  # Prioritize face over hand
                            detected_emoji = emoji_file
                            pose_type = 'hand'
                            # Position emoji at hand center
                            palm = hand_landmarks.landmark[0]
                            emoji_position = (int(palm.x * w), int(palm.y * h))
                            pose_name = next((p['name'] for p in detector.poses if p['file'] == emoji_file), "Unknown")

            # Create meme output (static background with emoji - SEPARATE from camera feed)
            meme_output = detector.create_meme_output(detected_emoji, pose_name)

            # Add small emoji indicator on camera feed (KEEP VIDEO FEED CLEAN)
            if detected_emoji and emoji_position:
                frame = detector.overlay_emoji(frame, detected_emoji, emoji_position, size=100)

            # Display info on camera feed
            cv2.putText(frame, f"STATE: {pose_name.upper()}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Mode: {mode_name}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Face: {'ON' if show_face else 'OFF'} | Hand: {'ON' if show_hands else 'OFF'}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Show both windows
            cv2.imshow('Camera Feed', frame)
            cv2.imshow('Emoji Output', meme_output)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_hands = not show_hands
                print(f"Hand tracking: {'ON' if show_hands else 'OFF'}")
            elif key == ord('f'):
                show_face = not show_face
                print(f"Face tracking: {'ON' if show_face else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram terminated.")


if __name__ == "__main__":
    main()
