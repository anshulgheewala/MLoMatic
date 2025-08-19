const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const router = express.Router();

const UPLOADS_DIR = process.env.UPLOADS_DIR || 'uploads';
const PYTHON_PATH = process.env.PYTHON_PATH || 'python';
const TRAIN_SCRIPT = process.env.TRAIN_SCRIPT || path.join(__dirname, '..', 'scripts', 'train_Model.py');

// Ensure upload dir exists
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

const upload = multer({ dest: UPLOADS_DIR });

router.post('/', upload.single('file'), (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'No file uploaded' });

  const { targetColumn, problemType, replaceColumn, findValue, replaceValue, cleaningRules } = req.body;

  // Parse selectedModels
  let selectedModels = [];
  try {
    selectedModels = JSON.parse(req.body.selectedModels || '[]');
  } catch (e) {
    return res.status(400).json({ error: 'Invalid selectedModels JSON' });
  }

  // Parse cleaningRules
  let parsedCleaningRules = [];
  try {
    parsedCleaningRules = cleaningRules ? JSON.parse(cleaningRules) : [];
    if (!Array.isArray(parsedCleaningRules)) {
      return res.status(400).json({ error: 'Cleaning rules must be an array.' });
    }
  } catch (err) {
    return res.status(400).json({ error: 'Invalid cleaningRules JSON' });
  }

  // Arguments for Python
  const args = [
    file.path,
    targetColumn,
    problemType,
    JSON.stringify(selectedModels),
    JSON.stringify(parsedCleaningRules)
  ];

  const pyProcess = spawn(PYTHON_PATH, [TRAIN_SCRIPT, ...args]);

  pyProcess.on('error', (err) => {
    console.error('❌ Failed to start subprocess.', err);
    return res.status(500).json({ error: 'Failed to start the training process. Check Python path.' });
  });

  let result = '';
  pyProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    console.error(`❌ Python Error: ${data}`);
  });

  pyProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: 'Failed to train model.' });
    }
    try {
      const parsedResult = JSON.parse(result.trim()); // Expect only JSON
      return res.status(200).json(parsedResult);
    } catch (e) {
      console.error('❌ JSON parse error', e);
      return res.status(500).json({ error: 'Failed to parse Python script output.' });
    }
  });
});

module.exports = router;



