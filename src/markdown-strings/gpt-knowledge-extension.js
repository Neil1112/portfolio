import RAGImg from '../assets/gpt-knowledge-extension/rag.jpg'

export const markdownContent = `# GPT Knowledge Extension

I extended the knowledge of GPT-3.5 using Retrieval Augmented Generation (RAG) method using Langchain and ChromaDB as Vector database.

[Try Demo](https://neil1112-umang-ai-app-wyvnyh.streamlit.app)

<br/>

![RAGImg](${RAGImg})


## Implementation
`

export const codeString = `import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import nltk
import nltk.data
from io import StringIO
import time
from streamlit_chat import message
import uuid

# local imports
from utils.chatInterface import initiateChat, getInput, getResponse


# # open ai key
# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# download nltk punkt
# download_dir = os.path.join('./', 'nltk_data')
# nltk.data.load(
#     os.path.join(download_dir, 'tokenizers/punkt/english.pickle')
# )

# with st.sidebar:
#     openai_api_key = st.text_input('OpenAI API Key')


st.title('Umang.ai')
st.write('A collection of NLP tools for the Umang.ai project.')

st.info('Explore different information from the sidebar. \n')
st.subheader('Things to try:')
st.write('- Upload a text file')
st.write('- Ask long and short queries on it.')
st.write('- Compare the results of these queries. It will help us develop the Prompt Template to fine tune the queries.')
st.write('- Ask for brief vs detailed answers.')
st.write('- Also compare the results with pure GPT-4. It can be accessed from the sidebar.')
# create a file uploader
uploaded_file = st.file_uploader("Upload a file", type="txt")


if uploaded_file is not None:
    # read byte data
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # convert to string
    stringio = StringIO(bytes_data.decode('utf-8'))
    # st.write(stringio)

    # read file as string
    document = stringio.read()
    # st.write(document)

    # split into chunks
    with st.spinner('Splitting Documents...'):
        time.sleep(1)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([document])

    st.info('Splitted into {} documents'.format(len(texts)))
    # st.info('Document 1 {}'.format(texts[0]))

    # embeddings
    with st.spinner('Creating Embeddings...'):
        time.sleep(1)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # create a vectorstore to use as index
    persist_directory = "db"
    with st.spinner('Creating Vectorstore...'):
        # db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        vectordb = Chroma.from_documents(texts, embeddings)

    # loading the vectorstore
    with st.spinner('Loading Vectorstore...'):
        time.sleep(1)
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # expose index in a retriever interface
    retriever = vectordb.as_retriever()

    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever, chain_type="stuff")

    # create the query input form
    # with st.form("query_input"):
    #     query = st.text_input("Enter your query here", key="query")
    #     submitted = st.form_submit_button("Submit")
    #     if submitted:
    #         with st.spinner("Generating answer..."):
    #             answer = qa.run(query)
    #             st.write(answer)

    # initiate chat
    initiateChat()


    # get input from user
    user_input = getInput()

    # get response
    if user_input is not None:
        getResponse(user_input, qa)

    # show messages
    # finalResponse(user_input)
`