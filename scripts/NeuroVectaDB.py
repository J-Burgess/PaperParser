import os
import glob
import argparse
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredFileLoader

# Create the parser
parser = argparse.ArgumentParser(description='Process some pdf and db paths.')

# Add the arguments
parser.add_argument('--pdf_path',
                    type=str,
                    help='the path to the directory of PDF files',
                    required=True)

parser.add_argument('--db_path',
                    type=str,
                    help='the path to the vector db persistence directory',
                    required=True)

# Parse the arguments
args = parser.parse_args()

pdf_directory_path = args.pdf_path
persist_directory = args.db_path


model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cuda'}
embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
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


