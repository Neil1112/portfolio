// src/App.js
import React, { useState } from 'react';
import Header from './Header.js';
import ProjectCategory from './ProjectCategory.js';
import NotesCategory from './NotesCategory.js';
import { project_categories, projects } from '../data/projects_and_categories.js';
import { notes_categories, notes } from '../data/notes_and_categories.js';
import { certificates } from '../data/certificates.js';

const Home = () => {

  const [showProjects, setShowProjects] = useState(true); // Initially, show projects
  const [showNotes, setShowNotes] = useState(false);
  const [showCertificates, setShowCertificates] = useState(false);

  // Define functions to handle button clicks
  const handleShowProjectsClick = () => {
    setShowProjects(true);
    setShowNotes(false);
    setShowCertificates(false);
  };

  const handleShowNotesClick = () => {
    setShowNotes(true);
    setShowProjects(false);
    setShowCertificates(false);
  };

  const handleShowCertificatesClick = () => {
    setShowCertificates(true);
    setShowProjects(false);
    setShowNotes(false);
  };

  return (
    <div className="min-h-screen text-white py-8 w-full">
      <div className="container mx-auto p-8">
        <Header/>

        <div className="mb-4 flex justify-center">
          <button
            className={`${
              showProjects ? 'bg-active text-white hover:bg-[#529E4E]' : 'bg-secondary text-gray-400 hover:bg-[#1D222B] hover:text-gray-200'
            } px-4 py-2 rounded`}
            onClick={handleShowProjectsClick}
          >
            Projects
          </button>
          {/* <button
            className={`${
              showNotes ? 'bg-active text-white hover:bg-[#529E4E]' : 'bg-secondary text-gray-400 hover:bg-[#1D222B] hover:text-gray-200'
            } px-4 py-2 rounded`}
            onClick={handleShowNotesClick}
          >
            Notes
          </button> */}
          <button
            className={`${
              showCertificates ? 'bg-active text-white hover:bg-[#529E4E]' : 'bg-secondary text-gray-400 hover:bg-[#1D222B] hover:text-gray-200'
            } px-4 py-2 rounded`}
            onClick={handleShowCertificatesClick}
          >
            Certificates
          </button>
        </div>


        {showProjects && (
          <div>
            <p>Total: {Object.keys(projects).length}</p>

            {project_categories.map((category) => (
              <ProjectCategory
                key={category.name}
                categoryName={category.name}
                projects={category.projects}
              />
            ))}
          </div>
        )}

        {showNotes && (
          <div >
            <p>Total: {Object.keys(notes).length}</p>
            {notes_categories.map((category) => (
              <NotesCategory
                key={category.name}
                categoryName={category.name}
                notes={category.notes}
              />
            ))}   
          </div>
        )}

        {showCertificates && (
          <div >
            <p>Total: {Object.keys(certificates).length}</p>
            <div className='grid grid-cols-1 md:grid-cols-2 gap-4 w-full'>
              {certificates.map((certificate) => (
                <div className='mb-8'>
                  <h2 className='text-2xl font-bold mb-4'>{certificate.name}</h2>
                  <img src={certificate.image} alt={certificate.name} />
                </div>
              ))}  
            </div> 
          </div>
        )}





        
      </div>
    </div>
  );
}

export default Home;
