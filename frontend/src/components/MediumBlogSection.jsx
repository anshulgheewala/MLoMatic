import React, { useState, useEffect } from "react";
import { supervisedArticles } from "./data";
import NavBar from "./NavBar";
import {
  Container,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  CardActions,
  Button,
  colors,
} from "@mui/material";

const MediumBlogSection = () => {
  const categories = Object.keys(supervisedArticles);
  const [selectedCategory, setSelectedCategory] = useState(categories[0]);

  const subtopics = supervisedArticles[selectedCategory]
    ? Object.keys(supervisedArticles[selectedCategory])
    : [];

  const [selectedAlgorithm, setSelectedAlgorithm] = useState("");

  useEffect(() => {
    if (subtopics.length > 0) {
      setSelectedAlgorithm(subtopics[0]);
    } else {
      setSelectedAlgorithm("");
    }
  }, [selectedCategory]);

  const articles =
    supervisedArticles[selectedCategory]?.[selectedAlgorithm] || [];

  return (
    <>
      <NavBar />
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom sx={{mb:4}}>
          Know Your Model
        </Typography>

        {/* Dropdowns in single line */}
        <Box
          display="flex"
          gap={3}
          flexWrap="wrap"
          mb={4}
          sx={{ alignItems: "center" }}
        >
          <FormControl sx={{ minWidth: 200, flex: 1 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              label="Category"
            >
              {categories.map((cat) => (
                <MenuItem key={cat} value={cat}>
                  {cat}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {subtopics.length > 0 ? (
            <FormControl sx={{ minWidth: 200, flex: 1 }}>
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={selectedAlgorithm}
                onChange={(e) => setSelectedAlgorithm(e.target.value)}
                label="Algorithm"
              >
                {subtopics.map((algo) => (
                  <MenuItem key={algo} value={algo}>
                    {algo}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ alignSelf: "center" }}
            >
              No algorithms found for this category.
            </Typography>
          )}
        </Box>

        <Box>
          {articles.length > 0 ? (
            articles.map((article, i) => (
              <Card
                key={i}
                sx={{
                  mb: 3,
                  borderRadius: 3,
                  boxShadow: 3,
                  p: 2, // added card padding
                  "&:hover": { boxShadow: 6 },
                }}
              >
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="h6" fontWeight="bold" gutterBottom>
                    {article.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    {article.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    By: {article.author}
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: "flex-end", pr: 2 }}>
                  <Button
                    size="small"
                    variant="contained"
                    color="primary"
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
      textTransform: "none",
      transition: "all 0.2s ease-in-out",
      "&:hover": {
        backgroundColor: "primary.dark", // keep it darker blue on hover
        color: "white", // keep text white
        transform: "translateY(-2px)", // slight lift
      },
    }}
                  >
                    Read More
                  </Button>
                </CardActions>
              </Card>
            ))
          ) : (
            <Typography variant="body1" color="text.secondary">
              No articles found for this algorithm.
            </Typography>
          )}
        </Box>
      </Container>
    </>
  );
};

export default MediumBlogSection;
