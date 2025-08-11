import cv2
import mediapipe as mp
import numpy as np
import os
import random
from skimage.metrics import structural_similarity as ssim
import time

# ========== Setup MediaPipe ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)
mp_draw = mp.solutions.drawing_utils

# Globals
last_x, last_y = None, None
current_color = (0, 0, 255)  # Default red
canvas = None
eraser_mode = False

# ===== Drawing =====
def draw_from_finger(x, y, drawing):
    global last_x, last_y, canvas, current_color, eraser_mode
    if not drawing:
        last_x, last_y = None, None
        return
    color = (255, 255, 255) if eraser_mode else current_color
    thickness = 30 if eraser_mode else 8
    if last_x is not None and last_y is not None:
        cv2.line(canvas, (x, y), (last_x, last_y), color, thickness=thickness)
    last_x, last_y = x, y

# ===== Gesture Detection =====
def is_index_drawing(landmarks, threshold=0.02):
    index_tip = landmarks.landmark[8]
    index_mcp = landmarks.landmark[5]  # base of index finger
    return (index_tip.y + threshold) < index_mcp.y

# ===== Load Random Emoji =====
def get_random_emoji():
    folder = "public/emojis"
    if not os.path.exists(folder):
        raise Exception("Folder 'public/emojis' not found")
    emojis = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    if not emojis:
        raise Exception("No PNG emojis found in public/emojis")
    return random.choice(emojis)

# ===== Preprocess for Matching =====
def preprocess_for_matching(img, out_size=300):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    if cv2.countNonZero(edges) == 0:
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = th

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    if cv2.countNonZero(edges) == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    coords = cv2.findNonZero(edges)
    x, y, w, h = cv2.boundingRect(coords)

    crop = edges[y:y + h, x:x + w]
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    top = (size - h) // 2
    left = (size - w) // 2
    square[top:top + h, left:left + w] = crop

    out = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return out

# ===== Compare Images =====
def compare_images(canvas_np, emoji_path, min_draw_pixels=2000):
    emoji_raw = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji_raw is None:
        raise FileNotFoundError(f"Emoji not found: {emoji_path}")

    if emoji_raw.ndim == 3 and emoji_raw.shape[2] == 4:
        alpha_mask = (emoji_raw[:, :, 3] > 0).astype(np.uint8) * 255
        emoji_bgr = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(emoji_raw, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        emoji_bgr = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)

    user_mask = preprocess_for_matching(canvas_np, out_size=300)
    emoji_mask = preprocess_for_matching(emoji_bgr, out_size=300)

    kernel = np.ones((5, 5), np.uint8)
    user_mask = cv2.morphologyEx(user_mask, cv2.MORPH_CLOSE, kernel)
    emoji_mask = cv2.morphologyEx(emoji_mask, cv2.MORPH_CLOSE, kernel)

    if np.count_nonzero(user_mask) < min_draw_pixels:
        return 0

    ssim_score = ssim(user_mask, emoji_mask, data_range=255)
    inter = np.logical_and(user_mask > 0, emoji_mask > 0).sum()
    union = np.logical_or(user_mask > 0, emoji_mask > 0).sum()
    iou = (inter / union) if union > 0 else 0.0

    contours_user, _ = cv2.findContours(user_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_emoji, _ = cv2.findContours(emoji_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_score = 0
    if contours_user and contours_emoji:
        shape_score = 1 - min(cv2.matchShapes(contours_user[0], contours_emoji[0],
                                              cv2.CONTOURS_MATCH_I1, 0.0), 1.0)

    final_score = (0.45 * ssim_score + 0.35 * iou + 0.2 * shape_score) * 100.0
    return int(round(max(0, min(100, final_score))))

# ===== Main =====
def main():
    global canvas, last_x, last_y, current_color, eraser_mode

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    if not ret:
        print("Camera not available")
        return

    # Match canvas to actual frame size
    frame_height, frame_width = frame.shape[:2]
    canvas = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)

    score_display = None
    timer_duration = 30
    reset_required = False

    emoji_filename = get_random_emoji()
    emoji_path = os.path.join("public", "emojis", emoji_filename)
    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji_img.shape[2] == 4:
        alpha = emoji_img[:, :, 3] / 255.0
        bgr = emoji_img[:, :, :3]
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        emoji_img = (bgr * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
    emoji_img = cv2.resize(emoji_img, (120, 120))
    start_time = time.time()

    cv2.namedWindow("🧠 Emoji Drawing Game", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("🧠 Emoji Drawing Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            x = int(index_tip.x * canvas.shape[1])
            y = int(index_tip.y * canvas.shape[0])
            drawing = is_index_drawing(hand_landmarks)
            if not reset_required:
                draw_from_finger(x, y, drawing)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        elapsed = time.time() - start_time
        remaining = max(0, int(timer_duration - elapsed))
        if remaining == 0 and score_display is None:
            score_display = compare_images(canvas, emoji_path)
            reset_required = True

        overlay = cv2.addWeighted(frame, 0.55, canvas, 0.45, 0)

        # --- UI improvements below ---

        # Top bar background
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 90), (20, 20, 20), -1)

        # Time
        cv2.putText(overlay, f"⏳ {remaining}s", (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 2, cv2.LINE_AA)

        # Score
        if score_display is not None:
            cv2.putText(overlay, f"🏆 {score_display}%", (200, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Tool mode indicator
        if eraser_mode:
            cv2.putText(overlay, "🩹 Eraser Selected", (420, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(overlay, (410, 25), (760, 75), (200, 0, 0), 2)
        else:
            cv2.putText(overlay, "✏️ Drawing", (420, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, current_color, 2, cv2.LINE_AA)
            cv2.rectangle(overlay, (410, 25), (680, 75), current_color, 2)

        # Target emoji in top right
        overlay[10:130, overlay.shape[1] - 130:overlay.shape[1] - 10] = emoji_img
        cv2.putText(overlay, "🎯 Target", (overlay.shape[1] - 180, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("🧠 Emoji Drawing Game", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not reset_required:
            canvas[:, :] = 255
            last_x, last_y = None, None
        elif key == ord('r'):
            canvas[:, :] = 255
            last_x, last_y = None, None
            emoji_filename = get_random_emoji()
            emoji_path = os.path.join("public", "emojis", emoji_filename)
            emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
            if emoji_img.shape[2] == 4:
                alpha = emoji_img[:, :, 3] / 255.0
                bgr = emoji_img[:, :, :3]
                white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
                emoji_img = (bgr * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
            emoji_img = cv2.resize(emoji_img, (120, 120))
            score_display = None
            reset_required = False
            start_time = time.time()
        elif key == ord('2'):
            current_color = (0, 255, 255)  # Yellow
        elif key == ord('3'):
            current_color = (0, 0, 0)      # Black
        elif key == ord('4'):
            current_color = (255, 0, 0)    # Blue
        elif key == ord('e'):
            eraser_mode = not eraser_mode
            print(f"✏️ Mode: {'Eraser' if eraser_mode else 'Draw'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
