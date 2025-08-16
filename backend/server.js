// const express = require('express');
// const multer = require('multer');
// const path = require('path');
// const fs = require('fs');
// const { exec } = require('child_process');
// const trainRoute = require('./routes/trainRoutes');
// import { fileURLToPath } from 'url';

// const app = express();
// const upload = multer({ dest: 'uploads/' });
// const cors = require('cors');
// app.use(cors());


// // Middleware
// app.use(express.json());

// // Serve generated reports
// app.use('/report', express.static(path.join(__dirname, 'reports')));

// // Route: Check server status
// app.get('/', (req, res) => {
//   res.send('✅ Backend is running');
// });

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// app.get("/download-model/:filename", (req, res)=>{
//   const filePath = path.join(__dirname, "uploads", req.params.filename);

//   if(fs.existsSync(filePath)){
//      res.download(filePath, req.params.filename, (err) => {
//       if (err) {
//         console.error("Download error:", err);
//         res.status(500).send("File download failed");
//       }
// });
//   }else{
//     res.status(404).send("File not found");
//   }
// });
// // Route: CSV upload and analysis
// app.post('/upload', upload.single('file'), (req, res) => {
//   if (!req.file) {
//     return res.status(400).send('No file uploaded.');
//   }

//   const csvPath = path.join(__dirname, req.file.path);
//   const reportDir = path.join(__dirname, 'reports');

//   // Ensure 'reports' directory exists
//   if (!fs.existsSync(reportDir)) {
//     fs.mkdirSync(reportDir);
//   }

//   const reportPath = path.join(reportDir, `${req.file.filename}.html`);

//   // Call Python script to analyze and generate report
//   const command = `python scripts/analyze.py "${csvPath}" "${reportPath}"`;

//   exec(command, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`❌ Python error: ${stderr}`);
//       return res.status(500).send('Error generating HTML report.');
//     }

//     console.log(`✅ HTML report generated: ${reportPath}`);

//     // Send URL to frontend
//     res.status(200).json({
//       message: 'Report generated successfully',
//       reportUrl: `/report/${req.file.filename}.html`,
//     });
//   });
// });

// app.use('/train', trainRoute);

// // Start server
// const PORT = 5000;
// app.listen(PORT, () => {
//   console.log(`✅ Server listening on http://localhost:${PORT}`);
// })


// BELOW is actual copy

// import express from 'express';
// import multer from 'multer';
// import path from 'path';
// import fs from 'fs';
// import { exec } from 'child_process';
// import trainRoute from './routes/trainRoutes.js'; // note the .js extension for ES modules
// import { fileURLToPath } from 'url';
// import cors from 'cors';
// import predictRoutes from './routes/predictRoutes.js'
// import 'dotenv/config';

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// const app = express();
// const upload = multer({ dest: 'uploads/' });

// app.use(cors());
// app.use(express.json());

// // Serve generated reports
// app.use('/report', express.static(path.join(__dirname, 'reports')));
// // Check server statuspyth
// app.get('/', (req, res) => {
//   res.send('✅ Backend is running');
// });

// // Download trained model
// app.get("/download-model/:filename", (req, res) => {
//   const filePath = path.join(__dirname, "uploads", req.params.filename);

//   if (fs.existsSync(filePath)) {
//     res.download(filePath, req.params.filename, (err) => {
//       if (err) {
//         console.error("Download error:", err);
//         res.status(500).send("File download failed");
//       }
//     });
//   } else {
//     res.status(404).send("File not found");
//   }
// });

// // Upload CSV and generate report
// app.post('/upload', upload.single('file'), (req, res) => {
//   if (!req.file) {
//     return res.status(400).send('No file uploaded.');
//   }

//   const csvPath = path.join(__dirname, req.file.path);
//   const reportDir = path.join(__dirname, 'reports');

//   if (!fs.existsSync(reportDir)) {
//     fs.mkdirSync(reportDir);
//   }

//   const reportPath = path.join(reportDir, `${req.file.filename}.html`);

//   const command = `python scripts/analyze.py "${csvPath}" "${reportPath}"`;

//   exec(command, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`❌ Python error: ${stderr}`);
//       return res.status(500).send('Error generating HTML report.');
//     }

//     console.log(`✅ HTML report generated: ${reportPath}`);

//     res.status(200).json({
//       message: 'Report generated successfully',
//       reportUrl: `/report/${req.file.filename}.html`,
//     });
//   });
// });

// app.use('/train', trainRoute);
// app.use('/predict', predictRoutes);

// // Start server
// const PORT = 5000;
// app.listen(PORT, () => {
//   console.log(`✅ Server listening on http://localhost:${PORT}`);
// });


import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { exec } from 'child_process';
import trainRoute from './routes/trainRoutes.js';
import { fileURLToPath } from 'url';
import cors from 'cors';
import predictRoutes from './routes/predictRoutes.js';
import 'dotenv/config';  // Load .env

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Use env variables
const UPLOADS_DIR = process.env.UPLOADS_DIR || 'uploads';
const REPORTS_DIR = process.env.REPORTS_DIR || 'reports';
const PYTHON_PATH = process.env.PYTHON_PATH || 'python';
const ANALYZE_SCRIPT = process.env.ANALYZE_SCRIPT || './scripts/analyze.py';
const PORT = process.env.PORT || 5000;

const upload = multer({ dest: UPLOADS_DIR });

app.use(cors({
  origin: "https://mlomatic-frontend.onrender.com"
}));
// app.use(cors());
app.use(express.json());

// Serve generated reports
app.use('/report', express.static(path.join(__dirname, REPORTS_DIR)));

// Check server status
app.get('/', (req, res) => {
  res.send('✅ Backend is running');
});

// Download trained model
app.get("/download-model/:filename", (req, res) => {
  const filePath = path.join(__dirname, UPLOADS_DIR, req.params.filename);

  if (fs.existsSync(filePath)) {
    res.download(filePath, req.params.filename, (err) => {
      if (err) {
        console.error("Download error:", err);
        res.status(500).send("File download failed");
      }
    });
  } else {
    res.status(404).send("File not found");
  }
});

// Upload CSV and generate report
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded.');
  }

  const csvPath = path.join(__dirname, req.file.path);
  const reportDir = path.join(__dirname, REPORTS_DIR);

  if (!fs.existsSync(reportDir)) {
    fs.mkdirSync(reportDir);
  }

  const reportPath = path.join(reportDir, `${req.file.filename}.html`);

  const command = `${PYTHON_PATH} ${ANALYZE_SCRIPT} "${csvPath}" "${reportPath}"`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`❌ Python error: ${stderr}`);
      return res.status(500).send('Error generating HTML report.');
    }

    console.log(`✅ HTML report generated: ${reportPath}`);

    res.status(200).json({
      message: 'Report generated successfully',
      reportUrl: `/report/${req.file.filename}.html`,
    });
  });
});

app.use('/train', trainRoute);
app.use('/predict', predictRoutes);

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server listening on http://localhost:${PORT}`);
});
