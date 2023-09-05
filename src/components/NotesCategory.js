// NotesCategory.js
import React from 'react';
import Notes from './Notes';

const NotesCategory = ({ categoryName, notes }) => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-4">
        {categoryName}
        <span class="text-xl text-center justify-center text-gray-400 bg-secondary font-bold rounded-full px-2 py-1 ml-2">{Object.keys(notes).length}</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full">
        {notes.map((notes) => (
          <Notes
            key={notes.title}
            id={notes.id}
            title={notes.title}
            description={notes.description}
            image={notes.image}
          />
        ))}
      </div>
    </div>
  );
};

export default NotesCategory;
