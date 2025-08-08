// src/App.jsx
import { useState } from "react";

function App() {
  const [gameStarted, setGameStarted] = useState(false);

  const handleStartGame = async () => {
    setGameStarted(true);

    // Optional: send a request to local Python server if you make one
    try {
      await fetch("http://localhost:5000/start"); // You can write a Python Flask API to handle this
    } catch (err) {
      console.error("Python game could not be triggered from React. Please run it manually.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-black text-white">
      <h1 className="text-4xl font-bold mb-6">🎨 Emoji Drawing Game</h1>
      <p className="mb-4">Use your index finger to draw the emoji shown on screen.</p>

      <button
        onClick={handleStartGame}
        className="bg-green-500 px-6 py-3 rounded-xl text-xl hover:bg-green-600 transition"
      >
        Start Game
      </button>

      {gameStarted && (
        <p className="mt-6 text-lg text-yellow-300">
          Please check the camera window running separately. Use C to clear, Enter to score, Q to quit.
        </p>
      )}
    </div>
  );
}

export default App;
