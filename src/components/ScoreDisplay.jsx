// src/components/ScoreDisplay.jsx
export default function ScoreDisplay({ score }) {
  return (
    <div className="mt-6 text-2xl font-bold text-green-400">
      🎯 Your Score: {score} / 100
    </div>
  );
}
