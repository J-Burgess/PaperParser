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

# Define your embedding model
embedding = OpenAIEmbeddings()
# Initialize the vector db
persist_directory = "/home/james/Desktop/Projects/paperparser/VectorDB"
vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#Now need to iteratively mine the pdfs into text and add embeddings into the vectorDB.
#Need an ID for each paper and optional metadata. Citation/Title?

def get_pdf_files(directory):
    # Create the file path pattern to match PDF files
    file_pattern = os.path.join(directory, "*.pdf")
    # Use glob to get a list of PDF files
    pdf_files = glob.glob(file_pattern)
    return pdf_files

def get_txt_files(directory):
    # Create the file path pattern to match PDF files
    file_pattern = os.path.join(directory, "*.txt")
    # Use glob to get a list of PDF files
    txt_files = glob.glob(file_pattern)
    return txt_files


def extract_text_from_pdf(pdf_path):
    """
    This function extracts text from the specified PDF file.
    """
    # Open the PDF file in read-binary mode.
    with open(pdf_path, 'rb') as file:
        # Create a PDF file reader object.
        pdf = PyPDF2.PdfReader(file)
        # Initialize an empty string to hold the extracted text.
        text = ''
        # Iterate through all the pages in the PDF file.
        for page_num in range(len(pdf.pages)):
            # Get a page object.
            page = pdf.pages[page_num]
            # Extract the text from the page.
            page_text = page.extract_text()
            # Add the extracted text to the overall text.
            text += page_text + '\n'
    # Return the extracted text.
    return text


# Specify the directory you want to search for PDF files
pdf_directory_path = "/home/james/Desktop/Projects/paperparser/Papers"

# Call the function to get the list of PDF files
pdf_files_list = get_pdf_files(pdf_directory_path)
print(pdf_files_list)
# Print the list of PDF files

#Convert pdfs into text.
for i in range(len(pdf_files_list)):
    #Parse into text and add to chroma db.
    text = extract_text_from_pdf(pdf_files_list[i])
    #Save text to file
    with open(f'{pdf_directory_path}/paper{i}.txt', 'w') as file:
        # Write the text to the file
        file.write(text)


#Now need to split text files into chunks and embed into vectordb.
txt_files = get_txt_files(pdf_directory_path)
for i in range(len(pdf_files_list)):
    loader = TextLoader(txt_files[i])
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)




