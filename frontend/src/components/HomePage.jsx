// src/pages/HomePage.js

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import {Dialog, DialogContent,  DialogTitle, IconButton} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ModelResults from './modelResults';
import NavBar from './NavBar';
// Import components from Material-UI
import {
  Box,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Typography,
  Chip,
} from '@mui/material';
import axios from 'axios';
import Loader from './Loader';
import Footer from './Footer';

// Define the models available for each problem type
const modelOptions = {
  classification: ['Logistic Regression', 'Random Forest', 'Support Vector Machine'],
  regression: ['Linear Regression', 'Ridge', 'Lasso'],
};

function HomePage() {
  const [file, setFile] = useState(null);
  const [headers, setHeaders] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [selectedModels, setSelectedModels] = useState([]);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [reportUrl, setReportUrl] = useState('');
  const [isLoading, setIsLoading]= useState(true);
  const [trainResults, setTrainResults] =  useState(null);
  const [isTraining, setIsTraining] = useState(false);

//  const onDrop = useCallback((acceptedFiles) => {
//   const uploadedFile = acceptedFiles[0];
//   setFile(uploadedFile);

//   // Use Papaparse to get the CSV headers
//   Papa.parse(uploadedFile, {
//     preview: 1, // We only need the first row to get headers
//     complete: (results) => {
//       // THE FIX IS HERE:
//       // If results.meta.fields is undefined, default to an empty array.
//       setHeaders(results.meta.fields || []);

//       // Reset selections when a new file is dropped
//       setTargetColumn('');
//       setSelectedModels([]);
//     },
//   });
// }, []);

const onDrop = useCallback((acceptedFiles) => {
  const uploadedFile = acceptedFiles[0];
  setFile(uploadedFile);

  // Use Papaparse to get the CSV headers
  Papa.parse(uploadedFile, {
    // THE FIX IS HERE:
    header: true, // Tell papaparse the first row is the header
    
    preview: 1,   // We only need to read one row to get the headers
    complete: (results) => {
      // If results.meta.fields is undefined, default to an empty array.
      setHeaders(results.meta.fields || []);

      // Reset selections when a new file is dropped
      setTargetColumn('');
      setSelectedModels([]);
    },
  });
}, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleTrain = async() => {
    // This is where you'll trigger the backend API call
    
    if (!file) {
      alert("Please upload a file first");
      return;
    }
    
    setIsTraining(true);
    const formData = new FormData();

    formData.append('file', file);
    formData.append('targetColumn', targetColumn);
    formData.append('problemType', problemType);

    formData.append('selectedModels', JSON.stringify(selectedModels));

    setIsLoading(true);

    try{
      const response = await axios.post('http://localhost:5000/train', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setTrainResults(response.data);

      console.log('Server Response:', response.data);
      alert('Training request sent successfully! check the console.');
    }
    catch(err){
      console.error('❌ Error sending training request:', err);
      alert('An error occurred. Check the console for details.');
    }finally {
    setIsLoading(false);
    setIsTraining(false);
    setTimeout(() => setIsTraining(false), 500);
  }
  };
  
  const handleVisualize = async() => {

      if(!file){
        alert('Please Upload a file');
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      try{
        setIsVisualizing(true);
        const response = await axios.post('http://localhost:5000/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        const fullUrl = `http://localhost:5000${response.data.reportUrl}`;
        setReportUrl(fullUrl);
    }
    catch(err){
        console.error(err);
        alert('Error in Generating The report please check the console');
    }
    finally{
        setIsVisualizing(false);
    }
  }

  

  return (
    <>

    <NavBar></NavBar>
    
    {isTraining && <Loader />}
    <Box sx={{ p: 4, maxWidth: '800px', margin: 'auto' }}>
      
      <Typography variant="h4" component="h1" gutterBottom align="center" fontWeight="bold">
          Automated ML Platform
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Upload your data, configure your model, and get results in minutes.
        </Typography>

      {/* Step 1: File Upload */}
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed grey',
          padding: '20px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? '#e3f2fd' : 'transparent',
          mb: 3,
        }}
      >
        <input {...getInputProps()} />
        <p>Drag 'n' drop a CSV file here, or click to select a file</p>
        {file && <Chip label={file.name} onDelete={() => setFile(null)} />}
      </Box>

      {/* Step 2: Configuration (only shows after file upload) */}
      {file && (
        <>
          {/* Target Column Selection */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>Select Target Column</InputLabel>
            <Select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
            >
              {headers.map((header) => (
                <MenuItem key={header} value={header}>
                  {header}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Problem Type Selection */}
          <FormControl component="fieldset" sx={{ mb: 3 }}>
             <Typography component="legend">Select Problem Type</Typography>
            <RadioGroup
              row
              value={problemType}
              onChange={(e) => {
                setProblemType(e.target.value);
                setSelectedModels([]); // Reset models when type changes
              }}
            >
              <FormControlLabel value="classification" control={<Radio />} label="Classification" />
              <FormControlLabel value="regression" control={<Radio />} label="Regression" />
            </RadioGroup>
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
                  {selected.map((value) => (
                    <Chip key={value} label={value} />
                  ))}
                </Box>
              )}
            >
              <MenuItem value="best_one">
                <em>Select The Best One For Me</em>
              </MenuItem>
              {modelOptions[problemType].map((model) => (
                <MenuItem key={model} value={model}>
                  {model}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Step 3: Action Buttons */}
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleTrain}
              disabled={!targetColumn || selectedModels.length === 0}
            >
              Train Model
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleVisualize}
              disabled={isVisualizing || !file}
            >
              {isVisualizing ? "Generating..." : 'Get Visualization'}
            </Button>

          </Box>
          {trainResults && <ModelResults results= {trainResults}/>}
        </>
        
      )}
      <Dialog
      open={!!reportUrl}
      onClose={()=> setReportUrl('')}
      fullWidth
      maxWidth='lg'>
        <DialogTitle>
            <IconButton
            aria-label='close'
            onClick={()=> setReportUrl('')}
            sx={{
                position: 'absolute',
                right: 8,
                top: 8,
                color: (theme) => theme.palette.grey[500],
            }}>
                <CloseIcon></CloseIcon>
            </IconButton>
        </DialogTitle>
        <DialogContent>
            <iframe
            src={reportUrl}
            title='Data Visualization'
            width="100%"
            height='600px'
            style={{ border: 'none'}}
            />
        </DialogContent>
      </Dialog>
    </Box>

    <Footer></Footer>  
    </>
    
  );
}

export default HomePage;