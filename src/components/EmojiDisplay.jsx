// src/components/EmojiDisplay.jsx
export default function EmojiDisplay({ emoji }) {
  return (
    <div className="flex flex-col items-center justify-center">
      <h2 className="text-xl font-semibold mb-3">Draw This Emoji</h2>
      {emoji ? (
        <img
          src={`/emojis/${emoji}`}
          alt="Target emoji"
          className="w-32 h-32 border rounded-xl border-white shadow-lg"
        />
      ) : (
        <p>Loading emoji...</p>
      )}
    </div>
  );
}
