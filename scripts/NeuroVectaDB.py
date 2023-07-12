import os
import glob
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredFileLoader

model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cuda'}
embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)
vectordb = Chroma(persist_directory="/home/james/Desktop/Projects/paperparser/VectorDB/TestIterative", embedding_function=embedding)

# Initialize the vector db persistence location
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB/TestIterative"
# Specify the directory you want to search for PDF files
pdf_directory_path = "/home/james/Desktop/Projects/paperparser/Papers"

#Get a list of pdfs in the pdf directory
pdfs = glob.glob(pdf_directory_path + "/**/*.pdf", recursive=True)

#For each pdf, load it into the vector db
for pdf in pdfs:
    #Make a list of strings of the document and list of metadatas.
    strings = []
    metadatas = [] #Must be a list of dictionaries.
    loader = UnstructuredFileLoader(pdf)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    #For each of the texts, I need to clean the /n (and special characters potentially? TODO) out of them before embedding.
    print("Cleaning text of ", pdf, "...")
    for i in range(len(texts)):
        cleaned_string = texts[i].page_content.replace("\n", "")
        strings.append(cleaned_string)
        metadatas.append(texts[i].metadata)
    print("Embedding text of ", pdf, "...")
    vectordb.add_texts(texts=strings, metadatas=metadatas)


vectordb.persist()
# Clear from memory
vectordb = None


