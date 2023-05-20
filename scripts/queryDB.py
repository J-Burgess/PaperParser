from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


prompt_template = """
You are a helpful assistant that has been given access to a vector memory of scientific papers and 
tasked with returning summarized information and the citation based on the question given. 
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Always include the doi urls in your answer and the context you used to answer the question labelled with CONTEXT:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=vectordb.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs)

query = "Can you summarize for me some pros and cons of SpecGAN to perform single molecule spectrum imaging?"
print("QUERY: ", query)
print(qa.run(query))



