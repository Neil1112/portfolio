// Notes.js
import React from 'react';
import { Link } from 'react-router-dom';

const Notes = ({ id, title, description, image }) => {
  return (
    <div className="flex p-4 rounded-lg shadow-md mb-4 w-full bg-secondary">
      <img src={image} alt={title} className="w-16 h-16 mb-2" />
      <div className="flex flex-col ml-4">
        <h3 className="text-white text-xl font-semibold mb-2">{title}</h3>
        <p className="text-gray-400">{description}</p>
        <br/>
        <Link to={`/notes/${id}`} className="text-blue-600 hover:underline mt-auto">
          View Notes
        </Link>
      </div>
    </div>
  );
};

export default Notes;
