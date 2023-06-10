
import os
import glob
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import LlamaCppEmbeddings

#Testing with my local llama.cpp running model. Quantized to 4 bit to fit in memory.
embedding = LlamaCppEmbeddings(model_path="/home/james/Desktop/Projects/llama/llama.cpp/models/7B/ggml-model-q4_0.bin")
# Define your embedding model
#embedding = OpenAIEmbeddings()

# Initialize the vector db persistence location
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
# Specify the directory you want to search for PDF files
pdf_directory_path = "/home/james/Desktop/Projects/paperparser/Papers"


print("Loading documents...")
loader = DirectoryLoader(pdf_directory_path, glob="**/*.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


#For each of the texts, I need to clean the /n (and special characters potentially? TODO) out of them before embedding.
print("Cleaning text...")
for i in range(len(texts)):
    texts[i].page_content = texts[i].page_content.replace("\n", "")

print("Embedding text...")
vectordb = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
vectordb.persist()
#Clear from memory
vectordb = None


