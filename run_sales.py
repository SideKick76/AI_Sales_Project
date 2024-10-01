# %%
USE_LLM_LOCAL = True

# Local imports
import fetch_sales_data
import helper
import custom

# System imports
import shutil
import time
import os

from pprint import pprint  # Import the pprint module
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# AI-Modelimports
if USE_LLM_LOCAL:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings, GPT4AllEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    llm = ChatOllama(model="llama3.2", temperature=0,num_predict=2000, num_ctx=10000)   # argument to match the FAISS settings
    model_text_size=1500
else:
    from langchain_openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=3000) # argument to match the FAISS settings
    model_text_size=600


# %%
vector_path = "vector_local_" + str(USE_LLM_LOCAL)

# %%
# Example usage
process_erase_clicked = False

if process_erase_clicked:
    helper.erase_folder(vector_path)

# %%
def check_faiss_avaible():
    return  helper.check_folder_exists(vector_path)

def load_faiss():
    vectorstore_faiss =  FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore_faiss
# %%
def process_and_split_documents(verbose=False):
    doc = fetch_sales_data.read_text_files_to_docs()

    # Setup how to split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=model_text_size,  # Maximum size of a chunk
        chunk_overlap=20,  # Overlap between chunks
        length_function=len  # Function to measure chunk size
    )

    # Split text
    docs = text_splitter.split_documents(doc)
    if verbose:
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}:")
            print(f"sources: {doc.metadata}")
            print(f"Content: {doc.page_content}\n")
    return docs
# %%    

def create_and_save_faiss_index(docs):
    vectorstore_faiss = FAISS.from_documents(docs, embeddings)

    # Wait until the FAISS index is ready
    # time.sleep(2)

    # Save the FAISS index to a pickle file   
    vectorstore_faiss.save_local(vector_path)
    return vectorstore_faiss

# %%
Query = "Any SW development related to AI"


# %%
def search_faiss_index_with_score(vectorstore_faiss, query, k=4, fetch_k=100, filter=None, verbose=False):
    # Perform similarity search with score
    results = vectorstore_faiss.similarity_search_with_score(
        query,
        k=k,
        fetch_k=fetch_k,
        filter=filter
    )

    # Optionally, print the results if verbose is True
    if verbose:
        for res, score in results:
            print(f"\n\n->-Chunk->--> \n\n* {res.page_content} {res.metadata} {score:.1f} [{len(res.page_content)}]")

    return results

# %%

def generate_custom_prompt_and_invoke_chain(results, Query):
    # Retrieve the custom prompt from environment variables with a default value
    custom_prompt_string = custom.MY_PROMPT
    print(custom_prompt_string)
    
    # Customize the prompt to reflect the CTO's perspective
    custom_prompt = ChatPromptTemplate.from_template(
        "{custom_prompt_string} the following question: {Query} to this {context}"
    )

    documents = [result[0] for result in results]
    sources = [result[0].metadata['sources'] for result in results]

    seen_sources = set()

    # Create and invoke the chain
    chain = create_stuff_documents_chain(llm, custom_prompt)
    result = chain.invoke({"custom_prompt_string": custom_prompt_string, "context": documents, "Query": Query})
    # print(result)

    # Function to remove folder path and file extension
    def clean_source(source):
        return os.path.splitext(os.path.basename(source))

    # Print the sources
    # print("\nSources:")
    for source in sources:
        clean_src = clean_source(source)
        if clean_src not in seen_sources:
            # print(clean_src[0])
            seen_sources.add(clean_src)

    return result, seen_sources

# %%
