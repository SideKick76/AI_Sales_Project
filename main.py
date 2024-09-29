# %%
USE_LLM_LOCAL = False

# Local imports
import fetch_sales_data
import helper

# System imports
import shutil
import time
import os
import streamlit as st
import sys

from pprint import pprint  # Import the pprint module
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Function to simulate processing
def process_input(prompt):
    import time
    time.sleep(2)  # Simulate a delay
    return f"Processed result for: {prompt}"




# Streamlit page configuration
st.set_page_config(
    page_title="Sales AI App",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def main() -> None:

    st.subheader("🧠 AI-Sales Tool", divider="gray", anchor=False)

    col1, col2 = st.columns([1.5, 2])

    with col1:
        # Add a headline
        st.header("Source Content")      
        # Add smaller descriptive text
        st.write("""You can propose contect by uploading it here.  
                 Note that it will not be automatically integrated.
                 """)

    with col1:
        # File uploader widget
        file_upload = col1.file_uploader("Upload a pdf, text or ppt file ↓", type=["txt", "ppt", "pdf"], accept_multiple_files=False)

        if file_upload is not None:
            # Define the path to the download folder
            download_folder = "download_folder"
            
            # Create the folder if it doesn't exist
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)
            
            # Save the uploaded file to the download folder
            file_path = os.path.join(download_folder, file_upload.name)
            with open(file_path, "wb") as f:
                f.write(file_upload.getbuffer())
            
            st.success(f"File saved to {file_path}")


    with col2:
        # Add a headline
        st.header("Sales tool")      
        # Add smaller descriptive text
        st.write("Here you can ask a question to the knowledge base.")

    with col2:
        message_container = st.container(height=500, border=True)
        prompt = st.chat_input("Enter a prompt here...", max_chars=100)

    if prompt:
        with message_container:
            st.write("Running...")

        # Process the input
        result = process_input(prompt)

        # Clear the message container and display the result
        with message_container:
            st.empty()  # Clear previous messages
            st.write(result)

if __name__ == "__main__":
    main()