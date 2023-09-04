import React from 'react';
import ReactMarkdown from 'react-markdown';
import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css'




const ProjectDetails = () => {

    const { id } = useParams();
    const [markdownContent, setMarkdownContent] = useState('');
    const [codeString, setCodeString] = useState('');

    useEffect(() => {
        async function fetchMarkdownContent() {
            try {
                const file = `../markdown-strings/${id}.js`
                console.log(file)
                
                const module = await import('../markdown-strings/gan-implementation-in-pytorch')
                console.log(module)
                setMarkdownContent(module.markdownContent);
                setCodeString(module.codeString);
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
            
            <SyntaxHighlighter language="python" style={atomOneDark} customStyle={{padding: '25px'}}>
                {codeString}
            </SyntaxHighlighter>

        </div>
    );
};

export default ProjectDetails;
