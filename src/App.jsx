import { useState, useRef, useEffect } from "react";
import WebcamCanvas from "./components/WebcamCanvas";

// List of emoji image filenames
const emojiList = [
"star.png"
];

function App() {
  const [targetEmoji, setTargetEmoji] = useState("");
  const [timer, setTimer] = useState(20); // 20 seconds to draw
  const [score, setScore] = useState(null);
  const canvasRef = useRef(null);

  // Select a random emoji when component loads
  useEffect(() => {
    const randomEmoji = emojiList[Math.floor(Math.random() * emojiList.length)];
    setTargetEmoji(randomEmoji);
  }, []);

  // Timer countdown
  useEffect(() => {
    if (timer === 0) {
      handleScoring();
      return;
    }

    const interval = setInterval(() => {
      setTimer((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [timer]);

  // Simulate score (replace with real compareImage logic)
  const handleScoring = async () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const drawnImage = canvas.toDataURL("image/png");

    // Here you would call your image comparison logic
    // For now we simulate score:
    const randomScore = Math.floor(Math.random() * 100) + 1;
    setScore(randomScore);
  };

  const handleRestart = () => {
    setTimer(20);
    setScore(null);
    const newEmoji = emojiList[Math.floor(Math.random() * emojiList.length)];
    setTargetEmoji(newEmoji);

    // Clear canvas
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center p-6 space-y-6">
      <h1 className="text-3xl font-bold">Emoji Drawing Game 🎨</h1>

      <div className="flex gap-10 items-center">
        {/* Emoji Display */}
        <div className="text-center space-y-2">
          <h2 className="text-lg font-semibold">Draw this:</h2>
          {targetEmoji && (
            <img
              src={`/emojis/${targetEmoji}`}
              alt="target emoji"
              className="w-24 h-24 mx-auto"
            />
          )}
        </div>

        {/* Webcam Canvas */}
        <WebcamCanvas setCanvasRef={(ref) => (canvasRef.current = ref)} />
      </div>

      {/* Timer and Score */}
      <div className="text-center space-y-2">
        {score === null ? (
          <p className="text-xl">⏱️ Time Left: {timer}s</p>
        ) : (
          <div className="space-y-2">
            <p className="text-2xl">✅ Your Score: <span className="font-bold">{score}/100</span></p>
            <button
              onClick={handleRestart}
              className="bg-white text-black px-4 py-2 rounded-md font-semibold"
            >
              Play Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
