// src/components/WebcamCanvas.jsx
import { useRef, useEffect } from "react";
import { setupHandTracking } from "../utils/handTracker";

export default function WebcamCanvas({ setCanvasRef }) {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#ffffff";

    let lastX = null;
    let lastY = null;

    const drawFromFinger = (x, y) => {
      if (lastX === null || lastY === null) {
        lastX = x;
        lastY = y;
        return;
      }

      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.stroke();

      lastX = x;
      lastY = y;
    };

    setupHandTracking(videoRef.current, canvasRef.current, drawFromFinger);
    setCanvasRef(canvasRef);

    // Reset last position when component unmounts
    return () => {
      lastX = null;
      lastY = null;
    };
  }, []);

  return (
    <div className="relative">
      <video ref={videoRef} autoPlay playsInline className="hidden" />
      <canvas
        ref={canvasRef}
        width={400}
        height={300}
        className="border border-white rounded-lg shadow-md"
      />
    </div>
  );
}
