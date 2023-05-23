import requests
import json
import os
import glob
import PyPDF2

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader


# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"

#Now need to iteratively mine the pdfs into text and add embeddings into the vectorDB.
#Need an ID for each paper and optional metadata. Citation/Title?

# Specify the directory you want to search for PDF files
pdf_directory_path = "/home/james/Desktop/Projects/paperparser/Papers"

loader = DirectoryLoader(pdf_directory_path, glob="**/*.pdf")
documents = loader.load()
print(len(documents))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)

#For each of the texts, I need to clean the /n and special characters out of them before embedding.
#Then I need to add them to the vectorDB.
special_chars = ["\n", "\t", "\r", "\x0b", "\x0c"]

print("Cleaning text...")
for i in range(len(texts)):
    texts[i].page_content = texts[i].page_content.replace("\n", "")
    #print(texts[i].page_content)


vectordb = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None


