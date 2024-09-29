# %%
USE_LLM_LOCAL = True

# Local imports
import fetch_sales_data
import helper

# System imports
import shutil
import time
import os

from pprint import pprint  # Import the pprint module
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
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
else:
    from langchain_openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0) # argument as "max_tokens=" to match the FAISS settings

# Take environment variables from .env (especially openai api key)
from dotenv import load_dotenv
load_dotenv()

# %%

print(os.getenv("URL1"))
print(os.getenv("MY_PROMPT"))
print(os.getenv("FILE_PATH4"))

# %%
vector_path = "vector_local_" + str(USE_LLM_LOCAL)

doc = fetch_sales_data.read_text_files_to_docs()


# %%
# Example usage
process_erase_clicked = False

if process_erase_clicked:
    helper.erase_folder(vector_path)



# %%
# Setup how to split text
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1500,  # Maximum size of a chunk
    chunk_overlap=20,  # Overlap between chunks
    length_function=len  # Function to measure chunk size
)

# Split text
docs = text_splitter.split_documents(doc)
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}:")
    print(f"sources: {doc.metadata}")
    # print(f"Content: {doc.page_content}\n")

# %%    
vectorstore_faiss = FAISS.from_documents(docs, embeddings)

# Wait until the FAISS index is ready
# time.sleep(2)

# Save the FAISS index to a pickle file   
vectorstore_faiss.save_local(vector_path)


# %%
Query = "Any SW development related to AI"


# %%

results = vectorstore_faiss.similarity_search_with_score(
    Query,
    k=4,
    fetch_k=100,
    filter = None #filter={"sources": "SigmaConnectivity.pdf"}
    )
# for res, score in results:
#     print(f"\n\n->-Chunk->--> \n\n* {res.page_content} {res.metadata} {score:.1f} [{len(res.page_content)}]")

# %%

# Retrieve the custom prompt from environment variables with a default value
custom_prompt_string = os.getenv("MY_PROMPT")
print(custom_prompt_string)
# Customize the prompt to reflect the CTO's perspective
custom_prompt = ChatPromptTemplate.from_template("{custom_prompt_string} the following question: {Query} to this {context}")

# %%

documents = [result[0] for result in results]
sources = [result[0].metadata['sources'] for result in results]

seen_sources = set()


chain = create_stuff_documents_chain(llm,custom_prompt)
result = chain.invoke({"custom_prompt_string":custom_prompt_string, "context": documents, "Query": Query})
print(result)

# Function to remove folder path and file extension
def clean_source(source):
    import os
    return os.path.splitext(os.path.basename(source))

# Print the sources
print("\nSources:")
for source in sources:
    clean_src = clean_source(source)
    if clean_src not in seen_sources:
        print(clean_src[0])
        seen_sources.add(clean_src)


# %%
