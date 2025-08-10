# 🎨 AI Emoji Sketch Game

A fun, real-time AI-powered drawing game using your laptop webcam.  
A target emoji appears on screen, and your goal is to draw it in the air using hand gestures.  
The AI will track your hand, capture your sketch, and compare it to the target emoji — giving you a similarity score based on accuracy.

---

## ✨ Features
- 📸 Real-time webcam-based hand tracking
- 🎯 Target emoji selection from OpenMoji set
- 🖌️ Draw in the air with fingertip tracking (no mouse needed)
- 🧠 AI comparison with **SSIM (Structural Similarity Index)**
- ⏱️ 30-second timer per round
- 🎨 Multiple color options (Red, Blue, Yellow, Black)
- 🏆 Scoring system out of 100

---

## 🛠 Tech Stack
- `Python`
- `MediaPipe`
- `NumPy`
- `OpenCV`
- `OpenMoji`
- `scikit-image`

---

## 🎮 How to Play
1. **Run the game**:
2. A target emoji will appear on screen.
3. Use your index finger in front of the webcam to draw the emoji in the air.
4. After 30 seconds, the game will:
5. Compare your sketch to the target emoji
6. Display your similarity score
7. Press R to start a new round.
   
## 📂 Project Structure  
 ┣ 📜 game.py              # Main game script<br>
 ┣ 📂 public/
 ┃   ┗ 📂 emojis           # Folder with target emojis<br>
 ┣ 📜 requirements.txt     # Python dependencies<br>
 ┗ 📜 README.md            # Project info<br>


## 📦 Installation
Clone this repository:
git clone https:<br>``https://github.com/PrathamRanka/emoji-recognition-game.git`` <br>
``cd emoji-sketch-game``   <br>
Install dependencies:
``pip install -r requirements.txt`` <br>
Make sure you have a webcam connected.<br>
Run the game: ``python emoji_draw_game.py``<br>


## 📝 License
This project is open source and available under the MIT License.
