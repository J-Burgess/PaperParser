# PaperParser (Work in progess)
Parse scientific PDF papers into summaries with an LLM and save as a persistent searchable vector memory.
Using langchain https://python.langchain.com/en/latest/index.html and the OpenAI API. 

# Long term goals:
* Iteratively build upon the database. Right now it bugs and overwrites the previous embeddings of the prior processed paper. 
* API access to Arxiv/Biorxiv/... to download papers based on query search. 


# Overview
1.) Set up the embedding model and vector database:
    Create an instance of the OpenAIEmbeddings class and assign it to the embedding variable.
    Specify a directory path (persist_directory) where the vector database (vectordb) will be stored.
    Create an instance of the Chroma class, passing the embedding model and the persist directory.

2.) Define text splitting configurations:
    Create an instance of the CharacterTextSplitter class with a specified chunk size and overlap.

3.) Define utility functions:
    get_pdf_files(directory): Given a directory, it uses the glob module to find PDF files in that directory and returns a list of file paths.
    get_txt_files(directory): Given a directory, it uses the glob module to find text files (with a .txt extension) in that directory and returns a list of file paths.
    extract_text_from_pdf(pdf_path): Given a PDF file path, it uses the PyPDF2 library to open the file, read its contents, and extract the text. The extracted text is returned as a string.

4.) Specify the directory containing PDF files:
    Assign the directory path to the pdf_directory_path variable.

5.) Get the list of PDF files:
    Call the get_pdf_files function with the pdf_directory_path and store the list of PDF file paths in the pdf_files_list variable.
    

6.) Convert PDFs to text and add to the vector database:
    Iterate over the PDF file paths in pdf_files_list.
    Extract text from each PDF file using the extract_text_from_pdf function.
    Save the extracted text to a corresponding text file in the pdf_directory_path.
    The text is then embedded using the embedding model and added to the vectordb using the Chroma.from_documents method.

7.) Persist the vector database:
    Save the vector database to disk using the persist method of the vectordb object.
    Set vectordb to None to release the resources.

8.) Query vector database:
    Using queryDB.py script and modifying the query variable you can search your vector database. 
   
![Example of query and response](https://github.com/J-Burgess/PaperParser/blob/main/Markdown_Journal/figures/screenshotA.png?raw=true)
