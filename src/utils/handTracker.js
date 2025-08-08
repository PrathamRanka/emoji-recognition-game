// src/utils/handTracker.js

import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

let drawingCallback = null;

export function setupHandTracking(videoElement, canvasElement, onFingerMove) {
  drawingCallback = onFingerMove;

  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });

  hands.onResults(onResults);

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480,
  });

  camera.start();
}

function onResults(results) {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Index finger tip is landmark 8
    const indexTip = landmarks[8];

    const x = indexTip.x * 640;
    const y = indexTip.y * 480;

    if (drawingCallback) {
      drawingCallback(x, y);
    }
  }
}
