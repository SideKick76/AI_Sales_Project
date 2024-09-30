# %%
USE_LLM_LOCAL = False

# Local imports
import run_sales

# System imports
import os
import streamlit as st

from pprint import pprint  # Import the pprint module
from langchain_community.vectorstores import FAISS

# Function to simulate processing
def process_input(prompt):
    if run_sales.check_faiss_avaible():
        vectorstore_faiss = run_sales.load_faiss()
    else:
        docs = run_sales.process_and_split_documents()
        vectorstore_faiss = run_sales.create_and_save_faiss_index(docs)
    results = run_sales.search_faiss_index_with_score(vectorstore_faiss, prompt)
    result, cleaned_sources = run_sales.generate_custom_prompt_and_invoke_chain(results,prompt)
    cleaned_sources = '\n'.join([f"- {item[0]}" for item in cleaned_sources])   # Make it cleaner and as bullets
    return f"{result} \n\nsources:\n{cleaned_sources}"


# Streamlit page configuration
st.set_page_config(
    page_title="Sales AI App",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def main() -> None:

    st.subheader("🧠 AI-Sales Tool, powered by {llm}", divider="gray", anchor=False)

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
        
        # "Clean All" button
        if st.button("Clear All"):
            # Logic to clear the content
            st.write("Content cleared!")
            with message_container:
                st.empty()
        
    if prompt:
        # Process the input
        answer = process_input(prompt)
        with col2:
            st.write(f"Your question was: {prompt}")
        # Clear the message container and display the result
        with message_container:
            st.empty()  # Clear previous messages
            st.write(answer)            
    
if __name__ == "__main__":
    main()