from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA

# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())
query = "What can you tell me about deep learning?"
print(qa.run(query))






