AI Sales Tool

This program is a Streamlit-based AI Sales Tool that processes user prompts to search and retrieve relevant information.
Default it run locally using ollama and llama3.2 model. For local run you need to run "ollama serve" and download LLM="llama3.2" and Embeddings="nomic-embed-text". 

Using Open AI chage USE_LLM_LOCAL=False in run_sales.py and set your key as described below.

Additional Setup
* Add Content
* Create a folder named "text\" and add your text files.

Environment Variables
Create a .env file in the project folder with the following content. 
Note that if you don’t have 7 text files, adjust fetch_sales_data.py accordingly:

* OPENAI_API_KEY=sk-.....
* MY_PROMPT="Answer as sales manager"
* FILE_PATH1=text/example1.txt
* FILE_PATH2=text/example2.txt
* FILE_PATH3=text/example3.txt
* FILE_PATH4=text/example4.txt
* FILE_PATH5=text/example5.txt
* FILE_PATH6=text/example6.txt
* FILE_PATH7=text/example7.txt
