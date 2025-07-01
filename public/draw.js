const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");

let drawing = false;

canvas.addEventListener("mousedown", (e) => {
  drawing = true;
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
});
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mouseleave", () => (drawing = false));

document.getElementById("clear-button").addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent = "";
});

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

export default canvas;
