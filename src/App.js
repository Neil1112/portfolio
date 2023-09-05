// App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home';
import ProjectDetails from './components/ProjectDetails';
import NotesDetails from './components/NotesDetails';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route exact path="/" Component={Home} />
        <Route path="/project/:id" Component={ProjectDetails} />
        <Route path="/notes/:id" Component={NotesDetails} />
      </Routes>
    </Router>
  );
};

export default App;
