from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
# Define your embedding model
model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cuda'}
embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

#embedding = OpenAIEmbeddings()

# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB/TestIterative"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

#initialize the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
#initialize the qa chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory,
                                           return_source_documents=True)


#Now I need to query the database with a question and a followup question.
query = "What can you tell me about single molecule spectrum imaging?"

result = qa({"question": query})
print("QUERY: ", query)
print("ANSWER: ", result["answer"])
print("SOURCE: ", result["source_documents"][0])


query = "What is a problem of BCI decoding?"
result = qa({"question": query})

print("QUERY: ", query)
print("ANSWER: ", result["answer"])
print("SOURCE: ", result["source_documents"][0])




