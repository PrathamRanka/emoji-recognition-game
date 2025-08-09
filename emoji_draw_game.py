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
    min_detection_confidence=0.95,
    min_tracking_confidence=0.95,
)
mp_draw = mp.solutions.drawing_utils

# Globals
last_x, last_y = None, None
current_color = (0, 0, 255)  # Default red
canvas = None
eraser_mode = False

# ===== Drawing =====
def draw_from_finger(x, y, drawing):
    global last_x, last_y, canvas, current_color
    if not drawing:
        last_x, last_y = None, None
        return
    if last_x is not None and last_y is not None:
        cv2.line(canvas, (x, y), (last_x, last_y),
                 current_color, thickness=20 if eraser_mode else 8)
    last_x, last_y = x, y

# ===== Gesture Detection =====
def is_pinch_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    dist = np.sqrt((thumb_tip.x - index_tip.x) ** 2 +
                   (thumb_tip.y - index_tip.y) ** 2)
    return dist < 0.05

def is_full_palm(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(landmarks.landmark[t].y < landmarks.landmark[p].y for t, p in zip(tips, pips))

# ===== Load Random Emoji =====
def get_random_emoji():
    folder = "public/emojis"
    if not os.path.exists(folder):
        raise Exception("Folder 'public/emojis' not found")
    emojis = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not emojis:
        raise Exception("No emojis found in public/emojis")
    return random.choice(emojis)

# ===== Scoring =====
def preprocess_for_matching(img, out_size=300):
    """
    Convert BGR image -> binary mask (edges / stroke mask), crop to content,
    center-pad to square, resize to out_size x out_size and return uint8 mask (0/255).
    """
    # grayscale + blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1) Try Canny edges (works well for strokes and emoji outlines)
    edges = cv2.Canny(blur, 50, 150)

    # 2) If Canny finds nothing (e.g., filled shapes or faint colors), fallback to Otsu-threshold
    if cv2.countNonZero(edges) == 0:
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = th

    # make strokes a bit thicker so small jitter doesn't kill matching
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # If no content at all → return blank mask
    if cv2.countNonZero(edges) == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    # crop to bounding rect of non-zero pixels
    coords = cv2.findNonZero(edges)
    x, y, w, h = cv2.boundingRect(coords)

    crop = edges[y:y + h, x:x + w]

    # pad to square (centered)
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    top = (size - h) // 2
    left = (size - w) // 2
    square[top:top + h, left:left + w] = crop

    # resize to fixed output size
    out = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return out


def compare_images(canvas_np, emoji_path, min_draw_pixels=250):
    """
    Returns score in percent (0-100).
    - canvas_np : BGR numpy array (your canvas)
    - emoji_path : path to target emoji (BGR read)
    - min_draw_pixels : minimum non-zero pixels required in user's drawing to be scored
    """
    # read emoji (handle alpha channel if present)
    emoji_raw = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji_raw is None:
        raise FileNotFoundError(f"Emoji not found: {emoji_path}")

    # If emoji has alpha, composite over white bg so we get proper edges
    if emoji_raw.ndim == 3 and emoji_raw.shape[2] == 4:
        alpha = (emoji_raw[:, :, 3] / 255.0).astype(np.float32)
        bgr = emoji_raw[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(bgr) * 255.0
        emoji_bgr = (bgr * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
    else:
        emoji_bgr = emoji_raw[:, :, :3] if emoji_raw.ndim == 3 else cv2.cvtColor(emoji_raw, cv2.COLOR_GRAY2BGR)

    # create comparable binary masks
    user_mask = preprocess_for_matching(canvas_np, out_size=300)
    emoji_mask = preprocess_for_matching(emoji_bgr, out_size=300)

    user_pixels = int(np.count_nonzero(user_mask))
    emoji_pixels = int(np.count_nonzero(emoji_mask))

    # If user drew almost nothing → 0 score
    if user_pixels < min_draw_pixels:
        return 0

    # Structural similarity on the binary masks (0..1)
    ssim_score = ssim(user_mask, emoji_mask, data_range=255)

    # Intersection over Union (IoU) between the masks (0..1)
    inter = np.logical_and(user_mask > 0, emoji_mask > 0).sum()
    union = np.logical_or(user_mask > 0, emoji_mask > 0).sum()
    iou = (inter / union) if union > 0 else 0.0

    # Final weighted score
    # I recommend heavier weight on SSIM but keep IoU to ensure overlap is required.
    final_score = (0.82 * ssim_score + 0.18 * iou) * 100.0
    final_score = max(0, min(100, final_score))

    return int(round(final_score))

# ===== Main =====
def main():
    global canvas, last_x, last_y, current_color, eraser_mode

    cap = cv2.VideoCapture(0)

    # Set to full screen resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas = 255 * np.ones((screen_height, screen_width, 3), dtype=np.uint8)

    score_display = None
    timer_duration = 30
    reset_required = False

    emoji_filename = get_random_emoji()
    emoji_path = os.path.join("public", "emojis", emoji_filename)
    emoji_img = cv2.imread(emoji_path)
    emoji_img = cv2.resize(emoji_img, (120, 120))
    start_time = time.time()

    print(f"🎯 Draw this emoji: {emoji_filename}")
    print("✍️ Pinch to draw, full palm to erase")

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

            if is_full_palm(hand_landmarks):
                eraser_mode = True
                current_color = (255, 255, 255)
            elif is_pinch_gesture(hand_landmarks):
                drawing = True
                if not eraser_mode:
                    current_color = (0, 0, 255)
            else:
                eraser_mode = False

            if not reset_required:
                draw_from_finger(x, y, drawing)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        elapsed = time.time() - start_time
        remaining = max(0, int(timer_duration - elapsed))

        if remaining == 0 and score_display is None:
            score_display = compare_images(canvas, emoji_path)
            print(f"✅ Time's up! Score: {score_display}%")
            reset_required = True

        overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        cv2.putText(overlay, f"Time: {remaining}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3)
        if score_display is not None:
            cv2.putText(overlay, f"Score: {score_display}%", (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        overlay[10:130, screen_width - 130:screen_width - 10] = emoji_img
        cv2.putText(overlay, "Target", (screen_width - 170, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.namedWindow("🧠 Emoji Drawing Game", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("🧠 Emoji Drawing Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("🧠 Emoji Drawing Game", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not reset_required:
            canvas[:, :] = 255
            last_x, last_y = None, None
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
