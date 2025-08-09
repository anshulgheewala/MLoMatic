import React from "react";
import { Box, Typography } from "@mui/material";
import MemoryIcon from "@mui/icons-material/Memory"; // Chip/ML-style icon

export default function Loader() {
  if (!open) return null;

  return (
    <Box
      sx={{
        position: "fixed",
        inset: 0,
        bgcolor: "rgba(0,0,0,0.85)",
        display: "flex",
        width: "100%",
        height: "100%",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        zIndex: 9999,
      }}
    >
      {/* Animated ML icon */}
      <MemoryIcon
        sx={{
          fontSize: 80,
          color: "#4cafef",
          animation: "spin 2s linear infinite",
          "@keyframes spin": {
            "0%": { transform: "rotate(0deg)" },
            "100%": { transform: "rotate(360deg)" },
          },
        }}
      />

      {/* Static text */}
      <Typography
        variant="h6"
        sx={{
          mt: 3,
          color: "#fff",
          fontWeight: "500",
        }}
      >
        Please wait while the model is training and tuning...
      </Typography>
    </Box>
  );
}
