const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const Jimp = require("jimp");
const app = express();
const PORT = 3000;
const modeljson = "file://./model/model.json";
const multer = require("multer");
const fs = require("fs");

const labels = ["The Fool", "The Magician", "The Chariot", "Temperence", "The Hierophant"];
/**
 * Process images and send to the tarot AI model for predictions
 * @param {*} imagePath - path of the image that is being uploaded to the tarot card model
 * @returns labeled predictions array sorted from most likely to least likely tarot card that matches the image
 */

let model;

(async () => {
  try {
    model = await tf.loadLayersModel(modeljson);
    console.log("✅ Model loaded successfully");
  } catch (e) {
    console.error("❌ Failed to load model", e);
  }
})();

async function classifyImage(imagePath) {
  // Variables
  let image = null
  let buffer = null
  let result = null
  // connect model

  // read image
  try {
    image = await Jimp.read(imagePath);
  } catch (e) {
    console.error("Failed to load image:", e);
    return 'Error: image load failure';
  }
 
  try {
    buffer = await new Promise((resolve, reject) => {
        image.getBuffer(Jimp.MIME_JPEG, (err, buf) => {
          if (err) reject(err);
          else resolve(buf);
        });
      });
  }
  catch (e) {
    console.error("Failed to load buffer:", e);
    return 'Error: buffer load failure';
  }

  // create image tensor
  const tensor = tf.node
    .decodeImage(buffer)
    .resizeNearestNeighbor([224, 224]) // Match the input size to model
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims(0);

  if (!tensor) return 'Error: unable to create image tensor'

  // // Make prediction
  const prediction = model.predict(tensor);
  
  try {
    result = await prediction.data();
    // Map prediction scores to labels
    const labeledPredictions = Array.from(result).map((score, idx) => ({
        label: labels[idx],
        confidence: score,
    }));

    // Sort by confidence descending
    labeledPredictions.sort((a, b) => b.confidence - a.confidence);

    return labeledPredictions;
  }
  catch (e) {
    console.error("Failed to get result:", e);
    return 'Error: prediction load failure';
  }
 
}

// Configure multer for image upload
const upload = multer({ dest: "uploads/" });

// GET Test endpoint
app.get("/predict", (req, res) => {
  res.json({ message: "Hello from the backend!" });
});

// POST endpoint to accept image
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }
  console.log('File received:', req.file);

  try {
    const result = await classifyImage(req.file.path);
    if (result) {
        console.log('got result', result)
        // Clean up uploaded file
        fs.unlinkSync(req.file.path);
    
        res.json({ prediction: result });
    }
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error processing image" });
  }
});

app.post("/upload-test", upload.single("image"), (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }
    console.log("Image upload successful:", req.file);
    res.status(200).json({ message: "Image received", file: req.file });
  });


// Run Server
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
