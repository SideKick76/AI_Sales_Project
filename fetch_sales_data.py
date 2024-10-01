
import custom
import os
import pdfplumber

from langchain_community.document_loaders import UnstructuredURLLoader

# %%
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata



# # %%
def read_pdfs_to_docs():
    documents = []
    for file_path in custom.FILE_PATHS:
        file_name = file_path.split('/')[-1]  # Extract the file name from the file path
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                documents.append(Document(page_content=text, metadata={'sources': file_name}))
    return documents


# %%
def read_text_files_to_docs():
    """
    Reads and parses .txt files.

    Returns:
    list: A list of Document objects containing the content and metadata for each file.
    """

    documents = []
    for file_path in custom.FILE_PATHS:
        with open(file_path, 'r') as file:
            text = file.read()
            documents.append(Document(page_content=text, metadata={'sources': file_path}))
    return documents

# %%

def save_web_data_to_files(urls):

    # Load the data from the URLs
    loader = UnstructuredURLLoader(urls=custom.URLS)
    web_data = loader.load()

    # Save the content to text files
    for url, document in zip(urls, web_data):
        # Extract text content from the Document object
        text_content = document.page_content
        
        # Create a valid filename from the URL
        filename = url.replace("https://", "").replace("/", "_") + ".txt"
        
        # Write the content to the file
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text_content)

    print("Content saved to text files.")