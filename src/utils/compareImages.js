// src/utils/compareImage.js

export async function compareImage(userCanvas, emojiImagePath) {
  const ctx1 = userCanvas.getContext("2d");
  const userImageData = ctx1.getImageData(0, 0, userCanvas.width, userCanvas.height).data;

  const emojiImg = new Image();
  emojiImg.src = emojiImagePath;

  return new Promise((resolve) => {
    emojiImg.onload = () => {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = userCanvas.width;
      tempCanvas.height = userCanvas.height;
      const tempCtx = tempCanvas.getContext("2d");

      tempCtx.drawImage(emojiImg, 0, 0, tempCanvas.width, tempCanvas.height);
      const emojiImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height).data;

      let totalDiff = 0;
      for (let i = 0; i < userImageData.length; i += 4) {
        const rDiff = Math.abs(userImageData[i] - emojiImageData[i]);
        const gDiff = Math.abs(userImageData[i + 1] - emojiImageData[i + 1]);
        const bDiff = Math.abs(userImageData[i + 2] - emojiImageData[i + 2]);

        totalDiff += rDiff + gDiff + bDiff;
      }

      const maxDiff = userCanvas.width * userCanvas.height * 255 * 3;
      const similarityScore = 100 - (totalDiff / maxDiff) * 100;

      resolve(Math.max(0, Math.min(100, Math.round(similarityScore))));
    };
  });
}
