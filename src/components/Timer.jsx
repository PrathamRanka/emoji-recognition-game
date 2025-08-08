// src/components/Timer.jsx
import { useEffect, useState } from "react";

export default function Timer({ duration, onComplete }) {
  const [seconds, setSeconds] = useState(duration);

  useEffect(() => {
    if (seconds <= 0) {
      onComplete(); // Time's up
      return;
    }

    const timer = setInterval(() => {
      setSeconds((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [seconds]);

  return (
    <div className="mt-4 text-lg">
      ⏱️ Time Left: <span className="font-bold">{seconds}s</span>
    </div>
  );
}
