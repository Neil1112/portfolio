// ProjectCategory.js
import React from 'react';
import Project from './Project';

const ProjectCategory = ({ categoryName, projects }) => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-4">
        {categoryName}
        <span class="text-xl text-center justify-center text-gray-400 bg-secondary font-bold rounded-full px-2 py-1 ml-2">{Object.keys(projects).length}</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full">
        {projects.map((project) => (
          <Project
            key={project.title}
            id={project.id}
            title={project.title}
            description={project.description}
            image={project.image}
          />
        ))}
      </div>
    </div>
  );
};

export default ProjectCategory;
