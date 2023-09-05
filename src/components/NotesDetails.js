import React from 'react';
import ReactMarkdown from 'react-markdown';
import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css'




const NotesDetails = () => {

    const { id } = useParams();
    const [markdownContent, setMarkdownContent] = useState('');

    useEffect(() => {
        async function fetchMarkdownContent() {
            try {
                const module = await import(`../notes-strings/${id}.js`)
                console.log(module)
                setMarkdownContent(module.markdownContent);
            } catch (error) {
                console.log(error);
            }
        }
        fetchMarkdownContent();
      }, [id]);

    return (
        <div className="markdown-container">

            <ReactMarkdown 
                children={markdownContent} 
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
                className='markdown-content'
            />
            

        </div>
    );
};

export default NotesDetails;
