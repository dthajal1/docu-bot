# Docu-bot
A GenAI powered Q&A bot for your documentation.

## Architecture
Here is the basic architecture of the RAG model.
![RAG Architecture](/static/imgs/rag_architecture.png)

**backend/core.py**
This file contains the logic for Retrieval Augmentation Generation (RAG).

**ingestion.py**
This file contains the code for ingesting the documentation data. It reads the data from local file, splits into chunks, embeds the chunks as vectors and stores it in a pinecone database.

## Usage
