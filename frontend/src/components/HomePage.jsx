// import React, { useState, useCallback } from 'react';
// import { useDropzone } from 'react-dropzone';
// import Papa from 'papaparse';
// import { Dialog, DialogContent, DialogTitle, IconButton, InputLabel, Select, MenuItem, FormControl, RadioGroup, FormControlLabel, Radio, Typography, Chip, Box, Button, TextField } from '@mui/material';
// import CloseIcon from '@mui/icons-material/Close';
// import ModelResults from './ModelResults';
// import NavBar from './NavBar';
// import axios from 'axios';
// import Loader from './Loader';
// import Footer from './Footer';
// import { ToastContainer, toast } from 'react-toastify';
// import 'react-toastify/dist/ReactToastify.css';

// const API_BASE_URL = import.meta.env.VITE_API_URL;

// const modelOptions = {
//   classification: ['Logistic Regression', 'Random Forest', 'Support Vector Machine', 'AdaBoost', 'XGBoost', 'Decission Tree'],
//   regression: ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'Support Vector Machine', 'Decission Tree'],
// };

// function HomePage() {
//   const [file, setFile] = useState(null);
//   const [headers, setHeaders] = useState([]);
//   const [targetColumn, setTargetColumn] = useState('');
//   const [problemType, setProblemType] = useState('classification');
//   const [selectedModels, setSelectedModels] = useState([]);
//   const [isVisualizing, setIsVisualizing] = useState(false);
//   const [reportUrl, setReportUrl] = useState('');
//   const [isLoading, setIsLoading] = useState(true);
//   const [trainResults, setTrainResults] = useState(null);
//   const [isTraining, setIsTraining] = useState(false);
//   const [classificationColumns, setClassificationColumns] = useState([]);
//   const [regressionColumns, setRegressionColumns] = useState([]);
//   const [inputData, setInputData] = useState({});
//   const [prediction, setPrediction] = useState(null);
//   const [predicting, setPredicting] = useState(false);

//   // Data cleaning states
//   const [replaceColumn, setReplaceColumn] = useState('');
//   const [findValue, setFindValue] = useState('');
//   const [replaceValue, setReplaceValue] = useState('');
//   const [uniqueColumnValues, setUniqueColumnValues] = useState([]);
//   const [parsedData, setParsedData] = useState([]);

//   const handleInputChange = (field, value) => {
//     setInputData(prev => ({ ...prev, [field]: value }));
//   };

//   const handlePredict = async () => {
//     if (!trainResults || !trainResults.model_path) {
//       toast.error("Please train a model first!");
//       return;
//     }
//     setPredicting(true);
//     try {
//       const absoluteModelPath = `${trainResults.model_path}`;
//       const response = await axios.post(`${API_BASE_URL}/predict`, {
//         modelPath: absoluteModelPath,
//         inputData,
//       });
//       setPrediction(response.data.prediction);
//       toast.success("Prediction successful!");
//     } catch (error) {
//       console.error(error);
//       toast.error("Prediction failed!");
//     }finally{
//       setPredicting(false);
//     }
//   };

//   const analyzeColumns = (data, fields) => {
//     const classCols = [];
//     const regressCols = [];
//     const CLASSIFICATION_THRESHOLD = 20;
//     fields.forEach(field => {
//       const values = data.map(row => row[field]).filter(val => val != null && val !== '');
//       if (values.length === 0) return;
//       const isNumeric = values.every(val => !isNaN(parseFloat(val)) && isFinite(val));
//       if (isNumeric) {
//         const uniqueValues = new Set(values.map(v => parseFloat(v))).size;
//         if (uniqueValues <= CLASSIFICATION_THRESHOLD) {
//           classCols.push(field);
//         } else {
//           regressCols.push(field);
//         }
//       } else {
//         classCols.push(field);
//       }
//     });
//     return { classification: classCols, regression: regressCols };
//   };

//   const onDrop = useCallback((acceptedFiles) => {
//     const uploadedFile = acceptedFiles[0];
//     setFile(uploadedFile);
//     Papa.parse(uploadedFile, {
//       header: true,
//       skipEmptyLines: true,
//       complete: (results) => {
//         const fields = results.meta.fields || [];
//         const data = results.data;
//         setParsedData(data);
//         if (!fields || data.length === 0) {
//           toast.error("Failed to parse the CSV file. Please check the file format.");
//           return;
//         }
//         const { classification, regression } = analyzeColumns(data, fields);
//         setHeaders(fields);
//         setClassificationColumns(classification);
//         setRegressionColumns(regression);
//         setTargetColumn('');
//         setSelectedModels([]);
//       },
//       error: (err) => {
//         console.error("Papaparse Error:", err);
//         toast.error("Failed to parse the CSV file.");
//       }
//     });
//   }, []);

//   const { getRootProps, getInputProps, isDragActive } = useDropzone({accept: { "text/csv": [".csv"] },onDrop});

//   const handleReplaceColumnChange = (col) => {
//     setReplaceColumn(col);
//     if (!col) {
//       setUniqueColumnValues([]);
//       return;
//     }
//     const values = parsedData.map(row => row[col]).filter(v => v != null && v !== '');
//     const uniqueVals = Array.from(new Set(values));
//     setUniqueColumnValues(uniqueVals);
//     setFindValue('');
//     setReplaceValue('');
//   };

//   const handleTrain = async () => {
//     if (!file) {
//       toast.error("Please upload a file first");
//       return;
//     }
//     setIsTraining(true);
//     const formData = new FormData();
//     formData.append('file', file);
//     formData.append('targetColumn', targetColumn);
//     formData.append('problemType', problemType);
//     formData.append('selectedModels', JSON.stringify(selectedModels));
//     formData.append('replaceColumn', replaceColumn);
//     formData.append('findValue', findValue);
//     formData.append('replaceValue', replaceValue || '');
//     setIsLoading(true);
//     try {
//       const response = await axios.post(`${API_BASE_URL}/train`, formData, {
//         headers: { 'Content-Type': 'multipart/form-data' },
//       });
//       setTrainResults(response.data);
//       toast.success("Model Trained Successfully!");
//     } catch (err) {
//       console.error('❌ Error sending training request:', err);
//       toast.error("Something went wrong..");
//     } finally {
//       setIsLoading(false);
//       setIsTraining(false);
//       setTimeout(() => setIsTraining(false), 500);
//     }
//   };

//   const handleVisualize = async () => {
//     if (!file) {
//       toast.error("Please Upload a file");
//       return;
//     }
//     const formData = new FormData();
//     formData.append('file', file);
//     try {
//       setIsVisualizing(true);
//       const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
//         headers: { 'Content-Type': 'multipart/form-data' },
//       });
//       const fullUrl = `${API_BASE_URL}${response.data.reportUrl}`;
//       setReportUrl(fullUrl);
//     } catch (err) {
//       console.error(err);
//       toast.error("Error in Generating The report.");
//     } finally {
//       setIsVisualizing(false);
//     }
//   };

//   return (
//     <>
//       <NavBar />
//       <ToastContainer position="bottom-right" autoClose={3000} theme="colored" />
//       {isTraining && <Loader />}
//       <Box sx={{ p: 4, maxWidth: '800px', margin: 'auto' }}>
//         <Typography variant="h4" align="center" fontWeight="bold">Automated ML Platform</Typography>
//         <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
//           Upload your data, configure your model, and get results in minutes.
//         </Typography>

//         {/* Upload */}
//         <Box {...getRootProps()} sx={{
//           border: '2px dashed grey', padding: '20px', textAlign: 'center',
//           cursor: 'pointer', backgroundColor: isDragActive ? '#e3f2fd' : 'transparent', mb: 3,
//         }}>
//           <input {...getInputProps()} />
//           <p>Drag 'n' drop a CSV file here, or click to select a file</p>
//           {file && <Chip label={file.name} onDelete={() => setFile(null)} />}
//         </Box>

//         {file && (
//           <>
//             {/* Problem Type */}
//             <FormControl component="fieldset" sx={{ mb: 3 }}>
//               <Typography component="legend">Select Problem Type</Typography>
//               <RadioGroup row value={problemType} onChange={(e) => {
//                 setProblemType(e.target.value);
//                 setSelectedModels([]);
//               }}>
//                 <FormControlLabel value="classification" control={<Radio />} label="Classification" />
//                 <FormControlLabel value="regression" control={<Radio />} label="Regression" />
//               </RadioGroup>
//             </FormControl>

//             {/* Target Column */}
//             <FormControl fullWidth sx={{ mb: 3 }}>
//               <InputLabel>Select Target Column</InputLabel>
//               <Select
//                 value={targetColumn}
//                 onChange={(e) => setTargetColumn(e.target.value)}
//                 disabled={(problemType === 'classification' && classificationColumns.length === 0) ||
//                   (problemType === 'regression' && regressionColumns.length === 0)}
//               >
//                 {(problemType === 'classification' ? classificationColumns : regressionColumns).map((header) => (
//                   <MenuItem key={header} value={header}>{header}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>

//             {/* Model Selection */}
//             <FormControl fullWidth sx={{ mb: 3 }}>
//               <InputLabel>Select Models</InputLabel>
//               <Select
//                 multiple value={selectedModels}
//                 onChange={(e) => setSelectedModels(e.target.value)}
//                 renderValue={(selected) => (
//                   <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
//                     {selected.map((value) => <Chip key={value} label={value} />)}
//                   </Box>
//                 )}
//               >
//                 <MenuItem value="best_one"><em>Select The Best One For Me</em></MenuItem>
//                 {modelOptions[problemType].map((model) => (
//                   <MenuItem key={model} value={model}>{model}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>

//             {/* Data Cleaning */}
//             <Typography variant="h6" sx={{ mt: 0, mb: 2 }}>Data Cleaning (Optional)</Typography>
//             <FormControl fullWidth sx={{ mb: 2 }}>
//               <InputLabel>Column to Modify</InputLabel>
//               <Select value={replaceColumn} onChange={(e) => handleReplaceColumnChange(e.target.value)}>
//                 {classificationColumns.map((header) => (
//                   <MenuItem key={header} value={header}>{header}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>

//             <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
//               <FormControl sx={{ flex: 1 }}>
//                 <InputLabel>Find Value</InputLabel>
//                 <Select
//                   value={findValue}
//                   onChange={(e) => { setFindValue(e.target.value); setReplaceValue(''); }}
//                   disabled={!replaceColumn}
//                 >
//                   {uniqueColumnValues.map((val) => (
//                     <MenuItem key={val} value={val}>{String(val)}</MenuItem>
//                   ))}
//                 </Select>
//               </FormControl>

//               <FormControl sx={{ flex: 1 }}>
//                 <InputLabel>Replace With</InputLabel>
//                 <Select value={replaceValue} onChange={(e) => setReplaceValue(e.target.value)}>
//                   <MenuItem value="null">null</MenuItem>
//                 </Select>
//               </FormControl>
//             </Box>

//             {/* Buttons */}
//             <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
//               <Button variant="contained" color="primary" onClick={handleTrain}
//                 disabled={!targetColumn || selectedModels.length === 0} sx={{ flex: 1 }}>
//                 Train Model
//               </Button>
//               <Button variant="outlined" color="secondary" onClick={handleVisualize}
//                 disabled={isVisualizing || !file} sx={{ flex: 1 }}>
//                 {isVisualizing ? "Generating..." : 'Get Visualization'}
//               </Button>
//             </Box>

//             {trainResults && <ModelResults results={trainResults} />}

//             {/* Prediction Form
//             {trainResults && (
//               <Box
//                 sx={{
//                   mt: 4,
//                   p: 3,
//                   border: '1px solid #ddd',
//                   borderRadius: 2,
//                   boxShadow: 2,
//                   backgroundColor: '#fafafa'
//                 }}
//               >
//                 <Typography variant="h5" gutterBottom fontWeight="bold">
//                   Predict on New Data
//                 </Typography>

//                 <Box
//                   sx={{
//                     display: 'grid',
//                     gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
//                     gap: 2,
//                     mt: 2
//                   }}
//                 >
//                   {headers
//                     .filter((h) => h !== targetColumn)
//                     .map((col) => (
//                       <TextField
//                         key={col}
//                         label={col}
//                         variant="outlined"
//                         fullWidth
//                         size="small"
//                         placeholder={`Enter value for ${col}`}
//                         value={inputData[col] || ''}
//                         onChange={(e) => handleInputChange(col, e.target.value)}
//                       />
//                     ))}
//                 </Box>

//                 <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
//                   <Button
//                     variant="contained"
//                     color="primary"
//                     size="large"
//                     sx={{ borderRadius: 2, textTransform: 'none', px: 4 }}
//                     onClick={handlePredict}
//                     disabled={Object.keys(inputData).length === 0}
//                   >
//                     Predict
//                   </Button>
//                 </Box>

//                 {prediction && (
//                   <Box
//                     sx={{
//                       mt: 3,
//                       p: 2,
//                       border: '1px solid #ccc',
//                       borderRadius: 2,
//                       backgroundColor: 'white',
//                       boxShadow: 1
//                     }}
//                   >
//                     <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
//                       Prediction Result:
//                     </Typography>
//                     <pre
//                       style={{
//                         margin: 0,
//                         background: '#f5f5f5',
//                         padding: '8px',
//                         borderRadius: '4px',
//                         overflowX: 'auto'
//                       }}
//                     >
//                       The predicted {targetColumn} is {Array.isArray(prediction) ? prediction[0] : prediction}
//                     </pre>
//                   </Box>
//                 )}
//               </Box>
//             )} */}

//             {/* Prediction Form */}
// {trainResults && (
//   <Box 
//     sx={{ 
//       mt: 4, 
//       p: 3, 
//       border: '1px solid #ddd', 
//       borderRadius: 2, 
//       boxShadow: 2, 
//       backgroundColor: '#fafafa' 
//     }}
//   >
//     <Typography variant="h5" gutterBottom sx={{ fontWeight: "bold" }}>
//       Predict on New Data
//     </Typography>

//     <Box 
//       sx={{ 
//         display: 'grid', 
//         gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, 
//         gap: 2, 
//         mt: 2 
//       }}
//     >
//       {headers
//         .filter((h) => h !== targetColumn)
//         .map((col) => {
//           const isCategorical = classificationColumns.includes(col);
//           const values = Array.from(new Set(parsedData.map(row => row[col]).filter(v => v !== "" && v != null)));

//           return isCategorical ? (
//             <FormControl key={col} fullWidth size="small">
//               <InputLabel>{col}</InputLabel>
//               <Select
//                 value={inputData[col] || ""}
//                 onChange={(e) => handleInputChange(col, e.target.value)}
//               >
//                 {values.map((val, idx) => (
//                   <MenuItem key={idx} value={val}>{String(val)}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>
//           ) : (
//             <TextField
//               key={col}
//               label={col}
//               variant="outlined"
//               fullWidth
//               size="small"
//               placeholder={`Enter value for ${col}`}
//               value={inputData[col] || ""}
//               onChange={(e) => handleInputChange(col, e.target.value)}
//             />
//           );
//         })}
//     </Box>

//     <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
//       <Button 
//         variant="contained" 
//         color="primary" 
//         size="large" 
//         sx={{ borderRadius: 2, textTransform: 'none', px: 4 }}
//         onClick={handlePredict}
//         disabled={predicting || Object.keys(inputData).length === 0}
//       >
//         {predicting ? "Predicting..." : "Predict"}
//       </Button>
//     </Box>

//     {prediction && (
//       <Box 
//         sx={{ 
//           mt: 3, 
//           p: 2, 
//           border: '1px solid #ccc', 
//           borderRadius: 2, 
//           backgroundColor: 'white', 
//           boxShadow: 1 
//         }}
//       >
//         <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: "bold" }}>
//           Prediction Result:
//         </Typography>
//         <pre 
//           style={{ 
//             margin: 0, 
//             background: '#f5f5f5', 
//             padding: '8px', 
//             borderRadius: '4px', 
//             overflowX: 'auto' 
//           }}
//         >
//           The predicted {targetColumn} is {Array.isArray(prediction) ? prediction[0] : prediction}
//         </pre>
//       </Box>
//     )}
//   </Box>
// )}

//           </>
//         )}

//         {/* Report Dialog */}
//         <Dialog open={!!reportUrl} onClose={() => setReportUrl('')} fullWidth maxWidth='lg'>
//           <DialogTitle>
//             <IconButton aria-label='close' onClick={() => setReportUrl('')} sx={{ position: 'absolute', right: 8, top: 8 }}>
//               <CloseIcon />
//             </IconButton>
//           </DialogTitle>
//           <DialogContent>
//             <iframe src={reportUrl} title='Data Visualization' width="100%" height='600px' style={{ border: 'none' }} />
//           </DialogContent>
//         </Dialog>
//       </Box>
//       <Footer />
//     </>
//   );
// }

// export default HomePage;


import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  InputLabel,
  Select,
  MenuItem,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  Typography,
  Chip,
  Box,
  Button,
  TextField
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ModelResults from './ModelResults';
import NavBar from './NavBar';
import axios from 'axios';
import Loader from './Loader';
import Footer from './Footer';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const API_BASE_URL = import.meta.env.VITE_API_URL;

const modelOptions = {
  classification: [
    'Logistic Regression',
    'Random Forest',
    'Support Vector Machine',
    'AdaBoost',
    'XGBoost',
    'Decision Tree'
  ],
  regression: [
    'Linear Regression',
    'Ridge',
    'Lasso',
    'Random Forest',
    'Support Vector Machine',
    'Decision Tree'
  ],
};

function HomePage() {
  const [file, setFile] = useState(null);
  const [headers, setHeaders] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [selectedModels, setSelectedModels] = useState([]);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [reportUrl, setReportUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [trainResults, setTrainResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [classificationColumns, setClassificationColumns] = useState([]);
  const [regressionColumns, setRegressionColumns] = useState([]);
  const [inputData, setInputData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [parsedData, setParsedData] = useState([]);

  // Data cleaning rules
  const [cleaningRules, setCleaningRules] = useState([]);

  const handleInputChange = (field, value) => {
    setInputData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    if (!trainResults || !trainResults.model_path) {
      toast.error("Please train a model first!");
      return;
    }
    setPredicting(true);
    try {
      const absoluteModelPath = `${trainResults.model_path}`;
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        modelPath: absoluteModelPath,
        inputData,
      });
      setPrediction(response.data.prediction);
      toast.success("Prediction successful!");
    } catch (error) {
      console.error(error);
      toast.error(error.message || "Prediction failed!");
    } finally {
      setPredicting(false);
    }
  };

  const analyzeColumns = (data, fields) => {
    const classCols = [];
    const regressCols = [];
    const CLASSIFICATION_THRESHOLD = 20;

    fields.forEach(field => {
      const values = data.map(row => row[field]).filter(val => val != null && val !== '');
      if (values.length === 0) return;

      const isNumeric = values.every(val => !isNaN(parseFloat(val)) && isFinite(val));
      if (isNumeric) {
        const uniqueValues = new Set(values.map(v => parseFloat(v))).size;
        if (uniqueValues <= CLASSIFICATION_THRESHOLD) {
          classCols.push(field);
        } else {
          regressCols.push(field);
        }
      } else {
        classCols.push(field);
      }
    });

    return { classification: classCols, regression: regressCols };
  };

  const onDrop = useCallback((acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    setFile(uploadedFile);

    Papa.parse(uploadedFile, {
      header: true,
      skipEmptyLines: true,
      worker: true, // ✅ prevents UI freeze on large CSVs
      complete: (results) => {
        const fields = results.meta.fields || [];
        const data = results.data;
        setParsedData(data);

        if (!fields || data.length === 0) {
          toast.error("Failed to parse the CSV file. Please check the file format.");
          return;
        }

        const { classification, regression } = analyzeColumns(data, fields);
        setHeaders(fields);
        setClassificationColumns(classification);
        setRegressionColumns(regression);
        setTargetColumn('');
        setSelectedModels([]);
        setCleaningRules([]); // reset cleaning rules when new file uploaded
      },
      error: (err) => {
        console.error("Papaparse Error:", err);
        toast.error("Failed to parse the CSV file.");
      }
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "text/csv": [".csv"] },
    onDrop
  });

  const handleRuleChange = (index, field, value) => {
    const newRules = [...cleaningRules];
    newRules[index][field] = value;
    if (field === "replaceColumn") {
      const values = parsedData.map(row => row[value]).filter(v => v != null && v !== '');
      newRules[index].uniqueColumnValues = Array.from(new Set(values));
    }
    setCleaningRules(newRules);
  };

  const handleAddRule = () => {
    setCleaningRules(prev => [
      ...prev,
      { replaceColumn: '', findValue: '', replaceValue: '', uniqueColumnValues: [] }
    ]);
  };

  const handleTrain = async () => {
    if (!file) {
      toast.error("Please upload a file first");
      return;
    }
    if (!targetColumn || selectedModels.length === 0) {
      toast.error("Please select target column and model(s)");
      return;
    }

    setIsTraining(true);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('targetColumn', targetColumn);
    formData.append('problemType', problemType);
    formData.append('selectedModels', JSON.stringify(selectedModels));
    formData.append(
      'cleaningRules',
      JSON.stringify(cleaningRules.filter(r => r.replaceColumn)) // ✅ send only valid rules
    );

    try {
      const response = await axios.post(`${API_BASE_URL}/train`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setTrainResults(response.data);
      toast.success("Model Trained Successfully!");
    } catch (err) {
      console.error('❌ Error sending training request:', err);
      toast.error(err.message || "Something went wrong..");
    } finally {
      setIsLoading(false);
      setIsTraining(false);
    }
  };

  const handleVisualize = async () => {
    if (!file) {
      toast.error("Please Upload a file");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsVisualizing(true);
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const fullUrl = `${API_BASE_URL}${response.data.reportUrl}`;
      setReportUrl(fullUrl);
    } catch (err) {
      console.error(err);
      toast.error("Error in Generating The report.");
    } finally {
      setIsVisualizing(false);
    }
  };

  return (
    <>
      <NavBar />
      <ToastContainer position="bottom-right" autoClose={3000} theme="colored" />
      {(isTraining || isLoading) && <Loader />}

      <Box sx={{ p: 4, maxWidth: '800px', margin: 'auto' }}>
        <Typography variant="h4" align="center" fontWeight="bold">
          Automated ML Platform
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Upload your data, configure your model, and get results in minutes.
        </Typography>

        {/* Upload */}
        <Box {...getRootProps()} sx={{
          border: '2px dashed grey', padding: '20px', textAlign: 'center',
          cursor: 'pointer', backgroundColor: isDragActive ? '#e3f2fd' : 'transparent', mb: 3,
        }}>
          <input {...getInputProps()} />
          <p>Drag 'n' drop a CSV file here, or click to select a file</p>
          {file && (
            <Chip
              label={file.name}
              onDelete={() => {
                setFile(null);
                setHeaders([]);
                setParsedData([]);
                setTargetColumn('');
                setSelectedModels([]);
              }}
            />
          )}
        </Box>

        {file && (
          <>
            {/* Problem Type */}
            <FormControl component="fieldset" sx={{ mb: 3 }}>
              <Typography component="legend">Select Problem Type</Typography>
              <RadioGroup
                row
                value={problemType}
                onChange={(e) => {
                  setProblemType(e.target.value);
                  setSelectedModels([]);
                }}
              >
                <FormControlLabel value="classification" control={<Radio />} label="Classification" />
                <FormControlLabel value="regression" control={<Radio />} label="Regression" />
              </RadioGroup>
            </FormControl>

            {/* Target Column */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Select Target Column</InputLabel>
              <Select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                disabled={
                  (problemType === 'classification' && classificationColumns.length === 0) ||
                  (problemType === 'regression' && regressionColumns.length === 0)
                }
              >
                {(problemType === 'classification' ? classificationColumns : regressionColumns)
                  .map((header) => (
                    <MenuItem key={header} value={header}>{header}</MenuItem>
                  ))}
              </Select>
            </FormControl>

            {/* Model Selection */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Select Models</InputLabel>
              <Select
                multiple
                value={selectedModels}
                onChange={(e) => setSelectedModels(e.target.value)}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => <Chip key={value} label={value} />)}
                  </Box>
                )}
              >
                <MenuItem value="best_one"><em>Select The Best One For Me</em></MenuItem>
                {modelOptions[problemType].map((model) => (
                  <MenuItem key={model} value={model}>{model}</MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Data Cleaning */}
            <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>Data Cleaning (Optional)</Typography>
            {cleaningRules.map((rule, index) => (
              <Box key={index} sx={{ mb: 3, border: "1px solid #ddd", p: 2, borderRadius: 2 }}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Column to Modify</InputLabel>
                  <Select
                    value={rule.replaceColumn}
                    onChange={(e) => handleRuleChange(index, "replaceColumn", e.target.value)}
                  >
                    {headers.map((header) => (
                      <MenuItem key={header} value={header}>{header}</MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
                  <FormControl sx={{ flex: 1 }}>
                    <InputLabel>Find Value</InputLabel>
                    <Select
                      value={rule.findValue}
                      onChange={(e) => handleRuleChange(index, "findValue", e.target.value)}
                      disabled={!rule.replaceColumn}
                    >
                      {rule.uniqueColumnValues.map((val) => (
                        <MenuItem key={val} value={val}>{String(val)}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <FormControl sx={{ flex: 1 }}>
                    <InputLabel>Replace With</InputLabel>
                    <Select
                      value={rule.replaceValue}
                      onChange={(e) => handleRuleChange(index, "replaceValue", e.target.value)}
                    >
                      <MenuItem value="null">null</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              </Box>
            ))}
            <Button variant="outlined" onClick={handleAddRule} sx={{ mb: 3 }}>
              ➕ Add Another Rule
            </Button>

            {/* Train & Visualize Buttons */}
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleTrain}
                disabled={!targetColumn || selectedModels.length === 0}
                sx={{ flex: 1 }}
              >
                Train Model
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                onClick={handleVisualize}
                disabled={isVisualizing || !file}
                sx={{ flex: 1 }}
              >
                {isVisualizing ? "Generating..." : 'Get Visualization'}
              </Button>
            </Box>

            {/* Training Results */}
            {trainResults && <ModelResults results={trainResults} />}

            {/* Prediction Form */}
            {trainResults && (
              <Box sx={{ mt: 4, p: 3, border: '1px solid #ddd', borderRadius: 2, boxShadow: 2, backgroundColor: '#fafafa' }}>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: "bold" }}>
                  Predict on New Data
                </Typography>

                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, mt: 2 }}>
                  {headers.filter((h) => h !== targetColumn).map((col) => {
                    const isCategorical = classificationColumns.includes(col);
                    const values = Array.from(new Set(parsedData.map(row => row[col]).filter(v => v !== "" && v != null)));

                    return isCategorical ? (
                      <FormControl key={col} fullWidth size="small">
                        <InputLabel>{col}</InputLabel>
                        <Select
                          value={inputData[col] || ""}
                          onChange={(e) => handleInputChange(col, e.target.value)}
                        >
                          {values.map((val, idx) => (
                            <MenuItem key={idx} value={val}>{String(val)}</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    ) : (
                      <TextField
                        key={col}
                        label={col}
                        variant="outlined"
                        fullWidth
                        size="small"
                        placeholder={`Enter value for ${col}`}
                        value={inputData[col] || ""}
                        onChange={(e) => handleInputChange(col, e.target.value)}
                      />
                    );
                  })}
                </Box>

                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    sx={{ borderRadius: 2, textTransform: 'none', px: 4 }}
                    onClick={handlePredict}
                    disabled={predicting || Object.keys(inputData).length === 0}
                  >
                    {predicting ? "Predicting..." : "Predict"}
                  </Button>
                </Box>

                {prediction && (
                  <Box sx={{ mt: 3, p: 2, border: '1px solid #ccc', borderRadius: 2, backgroundColor: 'white', boxShadow: 1 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: "bold" }}>
                      Prediction Result:
                    </Typography>
                    <pre style={{ margin: 0, background: '#f5f5f5', padding: '8px', borderRadius: '4px', overflowX: 'auto' }}>
                      The predicted {targetColumn} is {Array.isArray(prediction) ? prediction[0] : prediction}
                    </pre>
                  </Box>
                )}
              </Box>
            )}
          </>
        )}

        {/* Report Dialog */}
        <Dialog open={Boolean(reportUrl)} onClose={() => setReportUrl('')} fullWidth maxWidth='lg'>
          <DialogTitle>
            <IconButton aria-label='close' onClick={() => setReportUrl('')} sx={{ position: 'absolute', right: 8, top: 8 }}>
              <CloseIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent>
            {reportUrl && (
              <iframe src={reportUrl} title='Data Visualization' width="100%" height='600px' style={{ border: 'none' }} />
            )}
          </DialogContent>
        </Dialog>
      </Box>

      <Footer />
    </>
  );
}

export default HomePage;

