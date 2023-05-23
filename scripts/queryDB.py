from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

#initialize the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
#initialize the qa chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory,
                                           return_source_documents=True)


#Now I need to query the database with a question and a followup question.
query = "What is SpecGAN?"

result = qa({"question": query})
print("QUERY: ", query)
print("ANSWER: ", result["answer"])
print("SOURCE: ", result["source_documents"][0])


query = "What is the difference between SpecGAN and GAN?"
result = qa({"question": query})

print("QUERY: ", query)
print("ANSWER: ", result["answer"])
print("SOURCE: ", result["source_documents"][0])




