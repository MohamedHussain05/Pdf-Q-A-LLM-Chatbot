import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


from langchain_core.chat_history  import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')


embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Starting a project
st.title('Chat With Your PDF powered by RAG')
st.write('Upload Your PDF')

api_key=st.text_input('Enter Your Groq api key',type='password' )

if api_key:
    
    llm=ChatGroq(api_key=api_key,model_name='Gemma2-9b-It')
    
    session_id=st.text_input('Session_id',value='default_session')
    
    if 'store' not in st.session_state:
        st.session_state.store={}
        
    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
        
    uploaded_files=st.file_uploader('Upload aPDF file',type='pdf',accept_multiple_files=False)
    
    if uploaded_files:
        documents=[]
        
        for uploaded_file in uploaded_files:
            temppdf='./temp.pdf'
            with open(temppdf,'wb') as file:
                file.write(uploaded_files.getvalue())
                file_name=uploaded_files.name
            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            
            splits=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
            split_docs=splits.split_documents(documents)
            
            vector_store=FAISS.from_documents(split_docs,embeddings)
            retriever=vector_store.as_retriever()
    
        #History Aware Chains
        contextualize_prompt=(
                                        "Given a chat history and the latest user question"
                                        "which might reference context in the chat history,"
                                        "formulate a standalone question which can be understood"
                                        "without the chat history. Do NOT answer the question,"
                                        "just reformulate if it needed and otherwise return it as is"    
                                    )
        prompt=ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}'),
            ]
        )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,prompt)
        
        #Question Answering chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you"
            "don't know. Use three sentences maximum and keep the"
            "answer concise."
            "\n\n"
            "{context}") 
        
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}'),
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        conversational_rag_chain=RunnableWithMessageHistory(
                                                                rag_chain,
                                                                get_session_history,
                                                                input_messages_key='input',
                                                                history_messages_key='chat_message',
                                                                output_messages_key='answer'
        )
        
        user_input=st.text_input('Ask Your Question')
        
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {'input':user_input,},
                config={
                    'configurable':{'session_id':session_id}
                }
            )
            
            st.write(st.session_state.store)
            st.success('Assitant',response['answer'])
            st.write('Chat History',session_history.messages)

else:
    st.warning('Hey Please Enter API of GROQ')