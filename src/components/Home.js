// src/App.js
import React from 'react';
import Header from './Header.js';
import Category from './Category.js';
import { categories, projects } from '../data/projects_and_categories.js';


const Home = () => {

  return (
    <div className="min-h-screen text-white py-8 w-full">
      <div className="container mx-auto p-8">
        <Header/>


        <p>Total: {Object.keys(projects).length}</p>

        {categories.map((category) => (
          <Category
            key={category.name}
            categoryName={category.name}
            projects={category.projects}
          />
        ))}
      </div>
    </div>
  );
}

export default Home;
