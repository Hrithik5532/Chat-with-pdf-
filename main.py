from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, field_validator
import uvicorn
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_fireworks import FireworksEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
api_key = os.getenv("FIREWORKS_API_KEY")
groq_api_key = os.getenv("GROQ_API")

app = FastAPI()

class Query(BaseModel):
    question: str

    @field_validator('question')
    def validate_question(cls, value):
        if not value:
            raise ValueError('Question cannot be empty')
        return value

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(contents)
        
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    embeddings = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    vectors = FAISS.from_documents(chunks, embeddings)
    os.remove("temp.pdf")
    
    # Save vectors in a global state (this is a simple example, for production consider a different approach)
    global vector_store
    vector_store = vectors
    
    return {"message": "PDF processed and vectors created"}

@app.post("/ask-question/")
async def ask_question(query: Query):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({'input': query.question})
    return {"answer": response['answer'], "context": response["context"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
