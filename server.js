const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const Jimp = require("jimp");
const app = express();
const PORT = 3000;
const modeljson = "file://./model/model.json";
const multer = require("multer");

/**
 * This file configures the TarotAI server
 * 
 * This server has one main endpoint /predict which allows users
 * to upload an image of a tarot card and calls a tensorflow AI 
 * model to get the most likely tarot card title associated with 
 * the given image
 */

const labels = [
  "The Fool",
  "The Magician",
  "The Chariot",
  "Temperence",
  "The Hierophant",
];
let model;

/* Initialize tensorFlow model */
(async () => {
  try {
    model = await tf.loadLayersModel(modeljson);
    console.log("Success: Model loaded successfully");
  } catch (e) {
    console.error("Error: Failed to load model", e);
  }
})();

/**
 * Image Processing and Classification Function
 * @param {*} imageBuffer the uploaded image's request file buffer 
 * @returns 
 */
async function classifyImage(imageBuffer) {
  /* Read image */
  const image = await Jimp.read(imageBuffer);
  image.resize(224, 224);

  /* Create image buffer */
  const buffer = await new Promise((resolve, reject) => {
    image.getBuffer(Jimp.MIME_JPEG, (err, buf) => {
      if (err) reject(err);
      else resolve(buf);
    });
  });

  /* Create image tensor */
  const tensor = tf.node
    .decodeImage(buffer)
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims(0);

  if (!tensor) return "Error: unable to create image tensor";

  /* Make prediction */
  const prediction = model.predict(tensor);
  const result = await prediction.data();

  /* Map prediction scores to labels */
  const labeledPredictions = Array.from(result).map((score, idx) => ({
    label: labels[idx],
    confidence: score,
  }));

  /* Dispose of tensor and prediction to fix mem leaks */
  tensor.dispose()
  prediction.dispose?.()

  /* Sort by confidence descending */
  labeledPredictions.sort((a, b) => b.confidence - a.confidence);

  return labeledPredictions;
}

/* Configure multer for image upload */
const storage = multer.memoryStorage();
const upload = multer({ storage });

/**
 * GET -- test endpoint
 */
app.get("/predict", (res) => {
  res.json({ message: "Hello from the Tarot AI Backend!" });
});

/**
 * POST -- endpoint to accept image
 */
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }
  try {
    const result = await classifyImage(req.file.buffer);
    if (result) {
      res.json({ prediction: result });
    }
  } catch (err) {
    res.status(500).json({ error: "Error processing image" });
  }
});

/* Run Server */
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Tarot AI Server Running on Port ${PORT}`);
});
