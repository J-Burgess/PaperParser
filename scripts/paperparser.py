import requests
import json
import os
from urllib.request import urlretrieve
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)


#Now need to iteratively mine the pdfs into text and add embeddings into the vectorDB.
#Need an ID for each paper and optional metadata. Citation/Title?




