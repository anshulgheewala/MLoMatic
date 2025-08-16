// // import React from "react";

// // export default function ModelResults({ results }) {
// //     if (!results || !results.best_model) {
// //         return (
// //             <div className="p-4 text-red-600 border border-red-300 rounded-lg bg-red-50">
// //                 No results to show. please train a model first.
// //             </div>
// //         );

// //     }

// //     const { best_model, metric_name, score, best_params, all_results } = results;

// //     return (
// //         <div className="p-6 space-y-6">
// //             {/* BestOne will be shown here */}
// //             <div className="bg-green-50 border border-green-300 rounded-lg shadow-md p-5">
// //                 <h2 className="text-2xl font-bold text-green-800 mb-2">Best Model: {best_model}</h2>
// //                 <p className="text-lg">
// //                     <span className="font-semibold">{metric_name}:</span>{" "}{score.toFixed(4)}</p>
// //                 {best_params && Object.keys(best_params).length > 0 && (
// //                     <div className="mt-2">
// //                         <p className="font-semibold">Best Parameters:</p>
// //                         <ul className="list-disc list-inside text-sm">
// //                             {Object.entries(best_params).map(([key, value]) => (
// //                                 <li key={key}>
// //                                     {key}: {value.toString()}
// //                                 </li>
// //                             ))}
// //                         </ul>
// //                     </div>
// //                 )}
// //             </div>
// //             {/* All models will be shown here */}
// //         <div className="overflow-x-auto">
// //         <table className="min-w-full border border-gray-300 rounded-lg shadow-sm">
// //           <thead className="bg-gray-100">
// //             <tr>
// //               <th className="px-4 py-2 border-b text-left">Model Name</th>
// //               <th className="px-4 py-2 border-b text-left">{metric_name}</th>
// //               <th className="px-4 py-2 border-b text-left">Best Parameters</th>
// //             </tr>
// //           </thead>
// //           <tbody>
// //             {Object.entries(all_results).map(([model, data]) => (
// //               <tr
// //                 key={model}
// //                 className={`${
// //                   model === best_model ? "bg-green-100" : "bg-white"
// //                 }`}
// //               >
// //                 <td className="px-4 py-2 border-b font-medium">{model}</td>
// //                 <td className="px-4 py-2 border-b">
// //                   {data.score !== undefined ? data.score.toFixed(4) : "N/A"}
// //                 </td>
// //                 <td className="px-4 py-2 border-b text-sm">
// //                   {data.best_params && Object.keys(data.best_params).length > 0
// //                     ? Object.entries(data.best_params)
// //                         .map(([key, value]) => `${key}: ${value}`)
// //                         .join(", ")
// //                     : "—"}
// //                 </td>
// //               </tr>
// //             ))}
// //           </tbody>
// //         </table>
// //       </div>
// //     </div>
// //     )
// // }

// import React from "react";
// import { Box, Button, Typography, Paper, Table, TableBody, TableCell, TableHead, TableRow } from "@mui/material";
// import axios from "axios";
// import { saveAs } from "file-saver";

// export default function ModelResults({ results }) {
//     if (!results) return null;

//     const { best_model, metric_name, score, best_params, all_results,  model_path } = results;
//     const handleDownload = async() => {
//        try {
//       const filename = model_path.split("/").pop(); // get just the file name
//       const response = await axios.get(`http://localhost:5000/download-model/${filename}`, {
//         responseType: "blob"
//       });

//       saveAs(response.data, filename);
//     } catch (err) {
//       console.error("Error downloading file", err);
//     }
//     }
//     return (
//         <Box sx={{ mt: 4 }}>
//             <Typography variant="h5" gutterBottom>
//                 Training Results
//             </Typography>

//             <Paper sx={{ p: 2, mb: 4 }}>
//                 <Typography variant="subtitle1">
//                     <strong>Best Model:</strong> {best_model || "N/A"}
//                 </Typography>
//                 <Typography variant="subtitle1">
//                     <strong>{metric_name}:</strong> {score ? score.toFixed(4) : "N/A"}
//                 </Typography>
//                 <Typography variant="subtitle1">
//                     <strong>Best Hyperparameters:</strong>{" "}
//                     {best_params && Object.keys(best_params).length > 0
//                         ? JSON.stringify(best_params)
//                         : "Default parameters"}
//                 </Typography>
//             </Paper>

//             <Typography variant="h6" gutterBottom>
//                 All Model Results:
//             </Typography>

//             <Table>
//                 <TableHead>
//                     <TableRow>
//                         <TableCell>Model Name</TableCell>
//                         <TableCell>CV Score</TableCell>
//                         <TableCell>Train Score</TableCell>
//                         <TableCell>Test Score</TableCell>
//                         <TableCell>Generalization Gap</TableCell>
//                         <TableCell>Best Params</TableCell>
//                     </TableRow>
//                 </TableHead>
//                 <TableBody>
//                     {all_results &&
//                         Object.entries(all_results).map(([modelName, res]) => (
//                             <TableRow key={modelName}>
//                                 <TableCell>{modelName}</TableCell>
//                                 <TableCell>{res.cv_score?.toFixed(4)}</TableCell>
//                                 <TableCell>{res.train_score?.toFixed(4)}</TableCell>
//                                 <TableCell>{res.test_score?.toFixed(4)}</TableCell>
//                                 <TableCell>{res.generalization_gap?.toFixed(4)}</TableCell>
//                                 <TableCell>
//                                     {res.best_params && Object.keys(res.best_params).length > 0
//                                         ? JSON.stringify(res.best_params)
//                                         : "Default"}
//                                 </TableCell>
//                             </TableRow>
//                         ))}
//                 </TableBody>
//             </Table>
//             <Box>
//                         {(<Button
//                           variant="contained"
//                           color="primary"
//                           sx={{mt: 2}}
//                           onClick={handleDownload}
//                         >
//                           Download Model (.pkl)
//                         </Button>
//                         )}
//                       </Box>
//         </Box>
        
//     );
// }




// This is change 




// import React, { useState } from "react";
// import {
//   Box, Button, Typography, Paper, Table, TableBody, TableCell,
//   TableHead, TableRow, TableContainer, Dialog, DialogTitle, DialogContent
// } from "@mui/material";
// import axios from "axios";
// import { saveAs } from "file-saver";

// export default function ModelResults({ results }) {
//   const [open, setOpen] = useState(false);
//   const [selectedMatrix, setSelectedMatrix] = useState(null);

//   if (!results) return null;

//   const { best_model, metric_name, score, best_params, all_results, model_path } = results;

//   const handleDownload = async () => {
//     try {
//       const filename = model_path.split("/").pop();
//       const response = await axios.get(
//         `https://mlomatic.onrender.com/download-model/${filename}`,
//         { responseType: "blob" }
//       );
//       saveAs(response.data, filename);
//     } catch (err) {
//       console.error("Error downloading file", err);
//     }
//   };

//   const handleOpenMatrix = (matrix) => {
//     setSelectedMatrix(matrix);
//     setOpen(true);
//   };

//   const handleCloseMatrix = () => {
//     setOpen(false);
//     setSelectedMatrix(null);
//   };

//   return (
//     <Box sx={{ mt: 4 }}>
//       <Typography variant="h5" gutterBottom>
//         Training Results
//       </Typography>

//       <Paper sx={{ p: 2, mb: 4 }}>
//         <Typography variant="subtitle1">
//           <strong>Best Model:</strong> {best_model || "N/A"}
//         </Typography>
//         <Typography variant="subtitle1">
//           <strong>{metric_name}:</strong> {score ? score.toFixed(4) : "N/A"}
//         </Typography>
//         <Typography variant="subtitle1">
//           <strong>Best Hyperparameters:</strong>{" "}
//           {best_params && Object.keys(best_params).length > 0
//             ? JSON.stringify(best_params)
//             : "Default parameters"}
//         </Typography>
//       </Paper>

//       <Typography variant="h6" gutterBottom>
//         All Model Results:
//       </Typography>

//       <TableContainer component={Paper}>
//         <Table>
//           <TableHead>
//             <TableRow>
//               <TableCell>Model Name</TableCell>
//               <TableCell>CV Score</TableCell>
//               <TableCell>Train Score</TableCell>
//               <TableCell>Test Score</TableCell>
//               <TableCell>Generalization Gap</TableCell>
//               <TableCell>Precision</TableCell>
//               <TableCell>Recall</TableCell>
//               <TableCell>F1 Score</TableCell>
//               <TableCell>Confusion Matrix</TableCell>
//               <TableCell>Best Params</TableCell>
//             </TableRow>
//           </TableHead>

//           <TableBody>
//             {all_results &&
//               Object.entries(all_results).map(([modelName, res]) => (
//                 <TableRow
//                   key={modelName}
//                   sx={modelName === best_model ? { backgroundColor: "#e0ffe0" } : {}}
//                 >
//                   <TableCell>{modelName}</TableCell>
//                   <TableCell>{res.cv_score?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>{res.train_score?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>{res.test_score?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>{res.generalization_gap?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>{res.precision?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>{res.recall?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>{res.f1_score?.toFixed(4) ?? "—"}</TableCell>
//                   <TableCell>
//                     {Array.isArray(res.confusion_matrix) ? (
//                       <Button
//                         size="small"
//                         variant="outlined"
//                         onClick={() => handleOpenMatrix(res.confusion_matrix)}
//                       >
//                         View
//                       </Button>
//                     ) : "—"}
//                   </TableCell>
//                   <TableCell>
//                     {res.best_params && Object.keys(res.best_params).length > 0
//                       ? Object.entries(res.best_params).map(([k, v]) => (
//                           <div key={k}>
//                             <strong>{k}:</strong> {v.toString()}
//                           </div>
//                         ))
//                       : "Default"}
//                   </TableCell>
//                 </TableRow>
//               ))}
//           </TableBody>
//         </Table>
//       </TableContainer>

//       <Dialog open={open} onClose={handleCloseMatrix} maxWidth="sm" fullWidth>
//         <DialogTitle>Confusion Matrix</DialogTitle>
//         <DialogContent>
//           {selectedMatrix ? (
//             <table style={{ borderCollapse: "collapse", margin: "auto" }}>
//               <tbody>
//                 {selectedMatrix.map((row, i) => (
//                   <tr key={i}>
//                     {row.map((cell, j) => (
//                       <td
//                         key={j}
//                         style={{
//                           border: "1px solid #ccc",
//                           padding: "8px",
//                           textAlign: "center",
//                         }}
//                       >
//                         {cell}
//                       </td>
//                     ))}
//                   </tr>
//                 ))}
//               </tbody>
//             </table>
//           ) : (
//             "No matrix available"
//           )}
//         </DialogContent>
//       </Dialog>

//       <Box>
//         <Button
//           variant="contained"
//           color="primary"
//           sx={{ mt: 2 }}
//           onClick={handleDownload}
//         >
//           Download Model (.pkl)
//         </Button>
//       </Box>
//     </Box>
//   );
// }

import React, { useState } from "react";
import {
  Box, Button, Typography, Paper, Table, TableBody, TableCell,
  TableHead, TableRow, TableContainer, Dialog, DialogTitle, DialogContent
} from "@mui/material";
import axios from "axios";
import { saveAs } from "file-saver";

const API_BASE_URL = import.meta.env.VITE_API_URL;

export default function ModelResults({ results }) {
  const [open, setOpen] = useState(false);
  const [selectedMatrix, setSelectedMatrix] = useState(null);

  if (!results) return null;

  const { best_model, metric_name, score, best_params, all_results, model_path } = results;

  const handleDownload = async () => {
    try {
      const filename = model_path.split("/").pop();
      const response = await axios.get(
        `${API_BASE_URL}/download-model/${filename}`,
        { responseType: "blob" }
      );
      saveAs(response.data, filename);
    } catch (err) {
      console.error("Error downloading file", err);
    }
  };

  const handleOpenMatrix = (matrix) => {
    setSelectedMatrix(matrix);
    setOpen(true);
  };

  const handleCloseMatrix = () => {
    setOpen(false);
    setSelectedMatrix(null);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Training Results
      </Typography>

      <Paper sx={{ p: 2, mb: 4 }}>
        <Typography variant="subtitle1">
          <strong>Best Model:</strong> {best_model || "N/A"}
        </Typography>
        <Typography variant="subtitle1">
          <strong>{metric_name}:</strong> {score ? score.toFixed(4) : "N/A"}
        </Typography>
        <Typography variant="subtitle1">
          <strong>Best Hyperparameters:</strong>{" "}
          {best_params && Object.keys(best_params).length > 0
            ? JSON.stringify(best_params)
            : "Default parameters"}
        </Typography>
      </Paper>

      <Typography variant="h6" gutterBottom>
        All Model Results:
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Model Name</TableCell>
              <TableCell>CV Score</TableCell>
              <TableCell>Train Score</TableCell>
              <TableCell>Test Score</TableCell>
              <TableCell>Generalization Gap</TableCell>
              <TableCell>Precision</TableCell>
              <TableCell>Recall</TableCell>
              <TableCell>F1 Score</TableCell>
              <TableCell>Confusion Matrix</TableCell>
              <TableCell>Best Params</TableCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {all_results &&
              Object.entries(all_results).map(([modelName, res]) => (
                <TableRow
                  key={modelName}
                  sx={modelName === best_model ? { backgroundColor: "#e0ffe0" } : {}}
                >
                  <TableCell>{modelName}</TableCell>
                  <TableCell>{res.cv_score?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>{res.train_score?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>{res.test_score?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>{res.generalization_gap?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>{res.precision?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>{res.recall?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>{res.f1_score?.toFixed(4) ?? "—"}</TableCell>
                  <TableCell>
                    {Array.isArray(res.confusion_matrix) ? (
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleOpenMatrix(res.confusion_matrix)}
                      >
                        View
                      </Button>
                    ) : "—"}
                  </TableCell>
                  <TableCell>
                    {res.best_params && Object.keys(res.best_params).length > 0
                      ? Object.entries(res.best_params).map(([k, v]) => (
                          <div key={k}>
                            <strong>{k}:</strong> {v.toString()}
                          </div>
                        ))
                      : "Default"}
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={open} onClose={handleCloseMatrix} maxWidth="sm" fullWidth>
        <DialogTitle>Confusion Matrix</DialogTitle>
        <DialogContent>
          {selectedMatrix ? (
            <table style={{ borderCollapse: "collapse", margin: "auto" }}>
              <tbody>
                {selectedMatrix.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td
                        key={j}
                        style={{
                          border: "1px solid #ccc",
                          padding: "8px",
                          textAlign: "center",
                        }}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            "No matrix available"
          )}
        </DialogContent>
      </Dialog>

      <Box>
        <Button
          variant="contained"
          color="primary"
          sx={{ mt: 2 }}
          onClick={handleDownload}
        >
          Download Model (.pkl)
        </Button>
      </Box>
    </Box>
  );
}
