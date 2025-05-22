const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const Jimp = require("jimp");
const app = express();
const PORT = 3000;
const modeljson = "file://./model/model.json";
const multer = require("multer");
const fs = require("fs");

const labels = ["The Fool", "The Magician", "The Chariot"];
/**
 * Process images and send to the tarot AI model for predictions
 * @param {*} imagePath - path of the image that is being uploaded to the tarot card model
 * @returns labeled predictions array sorted from most likely to least likely tarot card that matches the image
 */
async function classifyImage(imagePath) {
  // connect model
  const model = await tf.loadLayersModel(modeljson);

  // format image
  const image = await Jimp.read(imagePath);

  const buffer = await new Promise((resolve, reject) => {
    image.getBuffer(Jimp.MIME_JPEG, (err, buf) => {
      if (err) reject(err);
      else resolve(buf);
    });
  });

  // create image tensor
  const tensor = tf.node
    .decodeImage(buffer)
    .resizeNearestNeighbor([224, 224]) // Match the input size to model
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims(0);

  // // Make prediction
  const prediction = model.predict(tensor);
  const result = await prediction.data();

  // Map prediction scores to labels
  const labeledPredictions = Array.from(result).map((score, idx) => ({
    label: labels[idx],
    confidence: score,
  }));

  // Sort by confidence descending
  labeledPredictions.sort((a, b) => b.confidence - a.confidence);

  return labeledPredictions;
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

  try {
    const result = await classifyImage(req.file.path);

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({ prediction: result });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error processing image" });
  }
});


// Run Server
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
