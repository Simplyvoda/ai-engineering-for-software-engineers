import streamlit as st
import os
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# GET API KEY FROM ENV
# -----------------------------
# The shared .env lives in the Projects/AI folder, three levels up from this file
load_dotenv(Path(__file__).parent.parent.parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# UI IMAGE
# -----------------------------
image_filename = "Educative.png"
st.image(image_filename, use_container_width=True)


# -----------------------------
# PDF READER
# -----------------------------
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


# -----------------------------
# FORMAT DOCS
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# RAG PIPELINE
# -----------------------------
def generate_response(uploaded_files, query_text):

    if not uploaded_files:
        return "Please upload at least one file."

    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not found in environment variables."

    # Extract text from every file, keeping each file's name
    documents = []
    file_names = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            documents.append(read_pdf(uploaded_file))
        else:
            documents.append(uploaded_file.read().decode("utf-8"))
        file_names.append(uploaded_file.name)

    # Split into chunks. Each chunk records which file it came from
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.create_documents(
        documents,
        metadatas=[{"source": name} for name in file_names]
    )

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY
    )

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # Vector DB
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # Prompt (no hub dependency)
    prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.

Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")

    # RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query_text)


# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more articles",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

query_text = st.text_input(
    "Enter your question:",
    placeholder="Ask something about the documents...",
    disabled=not uploaded_files
)

result = None

with st.form("myform", clear_on_submit=True, border=False):

    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_files and query_text)
    )

    if submitted:
        with st.spinner("Processing..."):
            result = generate_response(uploaded_files, query_text)

if result:
    st.info(result)