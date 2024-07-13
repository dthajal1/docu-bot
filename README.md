# Docu-bot
A GenAI powered Q&A bot for your documentation using LangChain, Pinecone and Streamlit.

![Docu-bot](/static/imgs/prototype.png)

## Backend Architecture
Here is the basic RAG architecture of our project.
![RAG Architecture](/static/imgs/rag_architecture.png)

**backend/core.py**
This file contains the logic for Retrieval Augmentation Generation (RAG).

**ingestion.py**
This file contains the code for ingesting the documentation data. It reads the data from local file, splits into chunks, embeds the chunks as vectors and stores it in a pinecone database.

## Frontend
The frontend is built using [Streamlit](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps). It is a simple UI that allows the user to ask questions and get answers from the bot.

## Environment Variables
Create a `.env` file in the project directory using `.env.example` as an example.

## Usage
Clone the repository:
```python
git clone git@github.com:dthajal1/docu-bot.git
```
Go to the project directory:
```python
cd docu-bot
```
Download/Move your documentation data to `/data` folder. Edit `ingestion.py` to read data from the folder.
```python
    # TODO: edit this
    path_to_docs = "path/to/your/docs" 
```
Install the dependencies:
```python
pipenv install
```
Ingest the documentation data:
```python
pipenv run python ingestion.py
```
Run the app (flask and streamlit):
```python
streamlit run main.py
```
Open the browser and go to `http://localhost:8501/`
