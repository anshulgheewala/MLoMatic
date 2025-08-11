import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import HomePage from './components/HomePage'
// import MediumBlogSection from './components/MediumBlogSection'
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import MediumBlogSection from './components/MediumBlogSection'
function App() {

  return (
    <Router>
        <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/blog" element={<MediumBlogSection />} />
      </Routes>
    </Router>
  )
}

export default App