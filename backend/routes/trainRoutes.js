// const express = require('express');
// const multer = require('multer')
// const {spawn} = require('child_process');
// const path = require('path')
// import 'dotenv/config';


// const router = express.Router();

// const upload = multer({dest: 'uploads/'});

// router.post('/', upload.single('file'), (req, res)=>{
//     const file = req.file;

//     const {targetColumn, problemType, replaceColumn, findValue, replaceValue} = req.body;

//     const selectedModels = JSON.parse(req.body.selectedModels);

//     // const pythonExecutable = "C:\\Users\\anshu\\Desktop\\mlf1older\\backend\\venv\\Scripts\\python.exe"; // Use your absolute path
//     const pythonExecutable = 'python3'; // Use your absolute path
//     const pythonScript = path.join(__dirname, '..', 'scripts', 'train_Model.py');

//     // console.log(findValue);
//     // console.log(replaceValue);
    
    

//     // Arguments passing

//     const args = [
//     file.path,
//     targetColumn,
//     problemType,
//     JSON.stringify(selectedModels), // Pass the array as a JSON string
//     replaceColumn || '',
//     findValue || '',
//     replaceValue || ''
//   ];

//   const pyProcess = spawn(pythonExecutable, [pythonScript, ...args]);

//   pyProcess.on('error', (err) => {
//   console.error('❌ Failed to start subprocess.', err);
//   return res.status(500).json({ error: 'Failed to start the training process. Check the python executable path.' });
// });

//    let result = '';
//   pyProcess.stdout.on('data', (data) => {
//     result += data.toString();
//   });

//   pyProcess.stderr.on('data', (data) => {
//     console.error(`❌ Python Error: ${data}`);
//   });

//   pyProcess.on('close', (code) => {
//     if (code !== 0) {
//       return res.status(500).json({ error: 'Failed to train model.' });
//     }
//     try {
//       // The Python script prints a JSON string, so we parse it
//       const parsedResult = JSON.parse(result);
//       return res.status(200).json(parsedResult);
//     } catch (e) {
//       return res.status(500).json({ error: 'Failed to parse Python script output.' });
//     }
//   });
// });

// module.exports = router;


const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
require('dotenv').config();  // load .env

const router = express.Router();

const UPLOADS_DIR = process.env.UPLOADS_DIR || 'uploads';
const PYTHON_PATH = process.env.PYTHON_PATH || 'python';
const TRAIN_SCRIPT = process.env.TRAIN_SCRIPT || path.join(__dirname, '..', 'scripts', 'train_Model.py');

const upload = multer({ dest: UPLOADS_DIR });

router.post('/', upload.single('file'), (req, res) => {
  const file = req.file;

  const { targetColumn, problemType, replaceColumn, findValue, replaceValue } = req.body;
  const selectedModels = JSON.parse(req.body.selectedModels);

  // Arguments to pass to Python
  const args = [
    file.path,
    targetColumn,
    problemType,
    JSON.stringify(selectedModels), // Pass array as JSON string
    replaceColumn || '',
    findValue || '',
    replaceValue || ''
  ];

  const pyProcess = spawn(PYTHON_PATH, [TRAIN_SCRIPT, ...args]);

  pyProcess.on('error', (err) => {
    console.error('❌ Failed to start subprocess.', err);
    return res.status(500).json({ error: 'Failed to start the training process. Check the python executable path.' });
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
      const parsedResult = JSON.parse(result); // Expect JSON output from Python
      return res.status(200).json(parsedResult);
    } catch (e) {
      return res.status(500).json({ error: 'Failed to parse Python script output.' });
    }
  });
});

module.exports = router;

