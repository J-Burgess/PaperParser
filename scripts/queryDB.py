from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)





qa = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(temperature=0.2), chain_type="stuff", retriever=vectordb.as_retriever())
result = qa({"question": "What is a popular approach for decoding BCIs?"}, return_only_outputs=True)

print("Answer: " + result["answer"].replace('\n', ' '))
print("Source: " + result["sources"])





