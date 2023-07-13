# NeuroVecta
Parse PDF papers into vector embeddings and save as a persistent vector database. The database can be queried with natural language upon which a semantic similarity search is carried out and relevant context returned to a conversational LLM chain which outputs its response, source and metadata tags.   

Designing a vector database for scientific papers instead of relying on relational databases is a strategic choice driven by the inherent high dimensionality of scientific papers. Unlike traditional relational databases, which excel at organizing structured data with fixed schemas, scientific papers often contain complex and unstructured information, making it challenging to represent them accurately in a tabular format. Through leveraging vector databases, researchers can efficiently handle the multidimensional nature of scientific papers. These databases excel at capturing and processing high-dimensional data, allowing for effective storage, retrieval, and analysis of scientific papers' intricate content, such as textual information, citations, author affiliations, and references. By utilizing a vector database in conjunction with a neural network Large Language Model (LLM) agent, researchers can dynamically engage with scientific papers, extracting context from this data to conduct natural language inquiries, thereby providing a novel method to unearth relevant sources and answers to their research questions in an intuitive and interactive manner.


Using langchain https://python.langchain.com/en/latest/index.html, Chroma vector database https://www.trychroma.com/ and the OpenAI API https://platform.openai.com/docs/introduction. 

# Requirements:
* Pip installed and use pip install -r requirements.txt to get all required software packages. 
* NVIDIA Cuda drivers must be installed for your system and a capable GPU to use local embed mode. 
  + Otherwise an openAI API key will be required to use their embedding service remotely. 

# Goals:
* Add paper title, authors and doi as metadata of the vector embeddings. Right now only outputs source document address of file. 
* API access to Arxiv/Biorxiv/... to download papers based on query search. 
* Containerize for portability.
* Multi-Modal embeddings support. 

# Example
* Embedded two papers from bioRxiv:
  + Darrel et al, Translating deep learning to neuroprosthetic control. DOI: https://doi.org/10.1101/2023.04.21.537581
  + Sha et al, Deep learning-enhanced single-molecule spectrum imaging. DOI: https://doi.org/10.1101/2023.05.08.539787

![Example of conversational queries and response](https://github.com/J-Burgess/PaperParser/blob/main/Markdown_Journal/figures/screenshotB.png?raw=true)
