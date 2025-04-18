from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate


import tempfile

import streamlit as st

#load_dotenv()

st.title('Document Question Answering BOT')
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

PdfFile = st.sidebar.file_uploader('Upload your PDF file here.', type='pdf')

if PdfFile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(PdfFile.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    print(docs)
    # with st.chat_message('ai'):
    #     for i, doc in enumerate(docs):
    #         st.markdown(f"**Page {i+1}:**")
    #         st.markdown(doc.page_content)
    
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs,embedding)
    print(vector_store)
    for doc_id, doc in vector_store.docstore._dict.items():
        print(f"ID: {doc_id}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
  

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 200,
    #     chunk_overlap = 0

    # )

    # result = splitter.split_documents(docs)
    # with st.chat_message('ai'):
    #     st.markdown(result)

model = ChatOpenAI(openai_api_key =openai_api_key)

if 'messages' not in st.session_state:
    st.session_state.messages =[]


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('What is up ?')
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})

    query_results = vector_store.similarity_search(prompt, k=3)

    with st.chat_message('ai'):
        content = "\n\n".join([doc.page_content for doc in query_results])
        template = PromptTemplate(
            template=""" Based on this content {content} give me answer of this question{query}
            """, input_variables=['content','query']
        )
        final_prompt = template.invoke({
            'content' : content,
            'query' : prompt
        })
        final_result = model.invoke(final_prompt) #need to write solid code here 
        st.markdown(final_result.content)

    st.session_state.messages.append({'role':'ai','content':final_result.content})   
                #     st.markdown(query.page_content)
        #     st.session_state.messages.append({'role':'ai','content': query.page_content})
    

    