import cv2
import mediapipe as mp
import numpy as np
import os
import random
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as ssim
import time

# ========== Setup MediaPipe ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

# ========== Globals ==========
canvas = None
last_x, last_y = None, None
current_color = (0, 0, 255)  # Default: Red

# ========== Drawing ==========
def draw_from_finger(x, y, drawing):
    global last_x, last_y, canvas, current_color
    if not drawing:
        last_x, last_y = None, None
        return
    if last_x is not None and last_y is not None:
        cv2.line(canvas, (x, y), (last_x, last_y), current_color, thickness=8)
    last_x, last_y = x, y

# ========== Gesture Detection ==========
def is_drawing_gesture(landmarks):
    index_tip = landmarks.landmark[8].y
    index_pip = landmarks.landmark[6].y

    middle_tip = landmarks.landmark[12].y
    middle_pip = landmarks.landmark[10].y

    ring_tip = landmarks.landmark[16].y
    ring_pip = landmarks.landmark[14].y

    pinky_tip = landmarks.landmark[20].y
    pinky_pip = landmarks.landmark[18].y

    thumb_tip_x = landmarks.landmark[4].x
    thumb_ip_x = landmarks.landmark[3].x

    index_up = index_tip < index_pip
    middle_down = middle_tip > middle_pip
    ring_down = ring_tip > ring_pip
    pinky_down = pinky_tip > pinky_pip
    thumb_closed = abs(thumb_tip_x - thumb_ip_x) < 0.03

    return index_up and middle_down and ring_down and pinky_down and thumb_closed

# ========== Load Random Emoji ==========
def get_random_emoji():
    folder = "public/emojis"
    if not os.path.exists(folder):
        raise Exception("Folder 'public/emojis' not found")
    emojis = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not emojis:
        raise Exception("No emojis found in public/emojis")
    return random.choice(emojis)

# ========== SSIM Score ==========
def compare_images(canvas_np, emoji_path):
    def preprocess_array(img_array):
        img = Image.fromarray(img_array).convert("L").resize((300, 300))
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = ImageOps.autocontrast(img)
        return np.array(img)

    def preprocess_file(img_path):
        img = Image.open(img_path).convert("L").resize((300, 300))
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = ImageOps.autocontrast(img)
        return np.array(img)

    user = preprocess_array(canvas_np)
    emoji = preprocess_file(emoji_path)

    if np.mean(user) > 245:
        return 0

    score, _ = ssim(user, emoji, full=True)
    return round(score * 100)

# ========== Main Game ==========
def main():
    global canvas, last_x, last_y, current_color

    screen_width = 640
    screen_height = 720

    canvas = 255 * np.ones((screen_height, screen_width, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    score_display = None
    timer_duration = 30
    reset_required = False

    emoji_filename = get_random_emoji()
    emoji_path = os.path.join("public", "emojis", emoji_filename)
    emoji_img = cv2.imread(emoji_path)
    emoji_img = cv2.resize(emoji_img, (120, 120))
    start_time = time.time()

    print(f"🎯 Draw this emoji: {emoji_filename}")
    print("✍️ Use index finger only to draw (others closed)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        drawing = False
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)

            drawing = is_drawing_gesture(hand_landmarks)

            if not reset_required and x > 0:
                # Only draw if finger is over canvas side (right half of full screen)
                draw_from_finger(x, y, drawing)

            color = (0, 255, 0) if drawing else (0, 0, 255)
            cv2.circle(frame, (x, y), 8, color, -1)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        elapsed = time.time() - start_time
        remaining = max(0, int(timer_duration - elapsed))

        if remaining == 0 and score_display is None:
            score_display = compare_images(canvas, emoji_path)
            print(f"✅ Time's up! Score: {score_display}%")
            reset_required = True

        # UI
        ui = frame.copy()
        cv2.rectangle(ui, (0, 0), (screen_width, 80), (0, 0, 0), -1)
        cv2.putText(ui, f"Time: {remaining}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3)
        if score_display is not None:
            cv2.putText(ui, f"Score: {score_display}%", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        # Emoji target
        ui[10:130, screen_width - 130:screen_width - 10] = emoji_img
        cv2.putText(ui, "🎯 Target", (screen_width - 170, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Current brush color
        cv2.circle(ui, (screen_width - 180, 50), 20, current_color, -1)

        # Combine webcam (left) + canvas (right)
        stacked = np.hstack((ui, canvas))

        # Optional: draw divider line between webcam and canvas
        divider_x = screen_width
        cv2.line(stacked, (divider_x, 0), (divider_x, screen_height), (0, 0, 0), thickness=4)

        if reset_required:
            cv2.putText(stacked, "Press R to Reset", (screen_width + 50, screen_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("🧠 Emoji Drawing Game", stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not reset_required:
            canvas[:, :] = 255
            last_x, last_y = None, None
            score_display = None
            start_time = time.time()
        elif key == ord('r') and reset_required:
            canvas[:, :] = 255
            last_x, last_y = None, None
            emoji_filename = get_random_emoji()
            emoji_path = os.path.join("public", "emojis", emoji_filename)
            emoji_img = cv2.imread(emoji_path)
            emoji_img = cv2.resize(emoji_img, (120, 120))
            score_display = None
            reset_required = False
            start_time = time.time()
        elif key == ord('1'):
            current_color = (0, 0, 255)  # Red
        elif key == ord('2'):
            current_color = (255, 0, 0)  # Blue
        elif key == ord('3'):
            current_color = (0, 255, 255)  # Yellow
        elif key == ord('4'):
            current_color = (0, 0, 0)  # Black

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
