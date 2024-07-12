from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    '''
    This function will load, split into chunks, embed as vectors and store the vectors in Pinecone.
    '''
    # Load the documents. 
    path_to_docs = "langchain-docs/api.python.langchain.com/en/latest" # TODO: edit this to the path of the docs you want your bot to ingest
    loader = ReadTheDocsLoader(path_to_docs)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    # Embed the chunks
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    print("****Storing to VectorStore done ***")


if __name__ == "__main__":
    ingest_docs()