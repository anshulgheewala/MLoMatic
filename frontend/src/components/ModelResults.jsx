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
                          <strong>{k}:</strong> {v != null ? v.toString() : "N/A"}
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
