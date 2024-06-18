from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os , shutil


embeddings_model = OpenAIEmbeddings()
path = "data/books"
DB_PATH = "data/chroma"

def load_docs():
    loader = DirectoryLoader(path,glob="*.md")
    document = loader.load()
    return document

def split_in_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(document)
    print(f"splitting done with {len(chunks)} chunks")
    return chunks

def store_chunks(chunks):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    db = Chroma.from_documents(chunks, embeddings_model, persist_directory = DB_PATH)
    db.persist()
    print('databse created')