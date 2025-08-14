import express from 'express';
import { spawn } from 'child_process';
import path from 'path';

const router = express.Router();

router.post('/', (req, res) => {
  const { modelPath, inputData } = req.body;

  if (!modelPath || !inputData) {
    return res.status(400).json({ error: 'Missing model path or input data' });
  }

  const pythonPath = process.env.PYTHON_PATH || 'C:\\Users\\anshu\\Desktop\\mlf1older\\backend\\venv\\Scripts\\python.exe';
  const predictScript = path.resolve('scripts', 'predict.py');

  const pyProcess = spawn(pythonPath, [predictScript, modelPath]);

  let stdout = '';
  let stderr = '';

  pyProcess.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  pyProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Prediction script exited with code ${code}`);
      console.error('stderr:', stderr);
      return res.status(500).json({ error: 'Prediction failed', details: stderr });
    }

    try {
      const output = JSON.parse(stdout);
      if (output.error) {
        return res.status(400).json({ error: output.error });
      }
      res.json(output);
    } catch (err) {
      console.error('Failed to parse prediction output:', err);
      res.status(500).json({ error: 'Failed to parse prediction output' });
    }
  });

  // Write inputData JSON to the python process stdin and close it
  pyProcess.stdin.write(JSON.stringify(inputData));
  pyProcess.stdin.end();
});

export default router;