const express = require('express');
const multer = require('multer')
const {spawn} = require('child_process');
const path = require('path')


const router = express.Router();

const upload = multer({dest: 'uploads/'});

router.post('/', upload.single('file'), (req, res)=>{
    const file = req.file;

    const {targetColumn, problemType} = req.body;

    const selectedModels = JSON.parse(req.body.selectedModels);

    const pythonExecutable = "C:\\Users\\anshu\\Desktop\\mlf1older\\backend\\venv\\Scripts\\python.exe"; // Use your absolute path
    const pythonScript = path.join(__dirname, '..', 'scripts', 'train_model.py');

    // Arguments passing

    const args = [
    file.path,
    targetColumn,
    problemType,
    JSON.stringify(selectedModels) // Pass the array as a JSON string
  ];

  const pyProcess = spawn(pythonExecutable, [pythonScript, ...args]);

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
      // The Python script prints a JSON string, so we parse it
      const parsedResult = JSON.parse(result);
      res.status(200).json(parsedResult);
    } catch (e) {
      res.status(500).json({ error: 'Failed to parse Python script output.' });
    }
  });
});

module.exports = router;



//   console.log('✅ Received training request:');
//   console.log('File:', file);
//   console.log('Target Column:', targetColumn);
//   console.log('Problem Type:', problemType);
//   console.log('Selected Models:', selectedModels);

//   if(!file || ! targetColumn || !problemType || !selectedModels){
//     return res.status(400).json({error: 'Missing required fields for training.'});
//   }

//   res.status(200).json({
//     message: 'Training request received successfully!',
//     config: req.body,
//   });
// });

// module.exports = router;