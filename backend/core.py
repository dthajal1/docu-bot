from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
import os
load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_llm(query: str):
    '''
    Retrieval + Generation. 
    This function takes a query, embeds it, retrieves the most similar documents 
    (to generate context), send it along with the query to the LLM model which 
    then generates well-crafted answers.
    '''
    # Load the embeddings and the Pinecone index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    doc_search = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

    # Load the chat LLM model
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Load the retrieval QA chat prompt. 
    # This prompt is combined with the retrieved documents (context) 
    # to generate the answer only based on the context.
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=doc_search.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])



''' Here is what the prompt looks like (hub.pull("langchain-ai/retrieval-qa-chat")) -- https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat:
    Answer any use questions based solely on the context below:

    <context>

    {context}

    </context>

    PLACEHOLDER

    chat_history
    HUMAN

    {input}
'''