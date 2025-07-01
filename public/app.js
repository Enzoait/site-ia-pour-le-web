import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";
import canvas from "./draw.js";

async function runModel() {
  const session = await ort.InferenceSession.create("./model.onnx");

  document
    .getElementById("submit-button")
    .addEventListener("click", async () => {
      
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = 28;
      tmpCanvas.height = 28;
      const tmpCtx = tmpCanvas.getContext("2d");
      tmpCtx.drawImage(canvas, 0, 0, 28, 28);

      const imageData = tmpCtx.getImageData(0, 0, 28, 28);
    
      function getBoundingBox(imgData) {
        let minX = 28,
          minY = 28,
          maxX = 0,
          maxY = 0;
        for (let y = 0; y < 28; y++) {
          for (let x = 0; x < 28; x++) {
            const idx = (y * 28 + x) * 4;
            if (imgData.data[idx] < 200) continue;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
        return { minX, minY, maxX, maxY };
      }

      const bbox = getBoundingBox(imageData);
      const boxW = bbox.maxX - bbox.minX + 1;
      const boxH = bbox.maxY - bbox.minY + 1;

      const centeredCanvas = document.createElement("canvas");
      centeredCanvas.width = 28;
      centeredCanvas.height = 28;
      const centeredCtx = centeredCanvas.getContext("2d");
      centeredCtx.fillStyle = "black";
      centeredCtx.fillRect(0, 0, 28, 28);

      const dx = Math.floor((28 - boxW) / 2);
      const dy = Math.floor((28 - boxH) / 2);

      centeredCtx.drawImage(
        tmpCanvas,
        bbox.minX,
        bbox.minY,
        boxW,
        boxH,
        dx,
        dy,
        boxW,
        boxH
      );

      const centeredImageData = centeredCtx.getImageData(0, 0, 28, 28);

      const input = new Float32Array(1 * 1 * 28 * 28);
      
      const mean = 0.5;
      const std = 0.5;

      for (let i = 0; i < 28 * 28; i++) {
        const pixel = 255 - centeredImageData.data[i * 4];
        const pixel01 = pixel / 255.0;
        let normalized = (pixel01 - mean) / std;
        input[i] = normalized;
      }

      const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
      const feeds = { input: tensor }; 

      try {
        const results = await session.run(feeds);
        const output = results.output.data; 

        console.log(session.inputNames);
        console.log(session.outputNames);

        let maxVal = -Infinity;
        let pred = -1;
        for (let i = 0; i < output.length; i++) {
          if (output[i] > maxVal) {
            maxVal = output[i];
            pred = i;
          }
        }
        function softmax(arr) {
          const array = Array.from(arr);
          const max = Math.max(...array);
          const exps = array.map((x) => Math.exp(x - max));
          const sum = exps.reduce((a, b) => a + b, 0);
          return exps.map((e) => e / sum);
        }

        const probs = softmax(output);
        const maxProb = Math.max(...probs);
        const predi = probs.indexOf(maxProb);

        const withIndex = probs.map((val, i) => ({ val, i }));

        withIndex.sort((a, b) => b.val - a.val);

        console.log("With index : ", withIndex)

        const sorted = [];

        withIndex.forEach((item, rank) => {
          sorted.push(item.i);
        });

        document.getElementById(
          "result"
        ).textContent = `Chiffre prédit : ${predi}. Autre prédictions possibles ${sorted.slice(1,4)} (confiance ${(
          maxProb * 100
        ).toFixed(1)}%)`;
      } catch (e) {
        document.getElementById(
          "result"
        ).textContent = `Erreur inference: ${e}`;
      }
    });
}

runModel();
