// App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home';
import ProjectDetails from './components/ProjectDetails';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route exact path="/" Component={Home} />
        <Route path="project" Component={ProjectDetails} />
      </Routes>
    </Router>
  );
};

export default App;
