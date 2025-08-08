```emoji-draw-game/
│
├── public/
│   └── emojis/            # Store emoji images here (like heart.png, smiley.png)
│
├── src/
│   ├── assets/            # Optional: for logos or backgrounds
│   ├── components/
│   │   ├── EmojiDisplay.jsx       # Shows the current emoji
│   │   ├── WebcamCanvas.jsx       # Captures webcam and draws with hand
│   │   ├── Timer.jsx              # Countdown timer
│   │   └── ScoreDisplay.jsx       # Shows score after game ends
│   │
│   ├── utils/
│   │   ├── handTracker.js         # MediaPipe setup and hand tracking logic
│   │   └── compareImages.js       # Compare drawn canvas vs emoji
│   │
│   ├── App.jsx            # Main layout + game logic
│   ├── main.jsx           # React entry
│   └── index.css          # Tailwind styles
│
└── tailwind.config.js
