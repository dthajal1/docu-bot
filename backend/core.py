from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
import os
load_dotenv()

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
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

    # along with regular retrieved documents (context) we also need to add context from our chat history
    # thus we create chat_history aware retriever
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase") 
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=doc_search.as_retriever(), prompt=rephrase_prompt
    )

    # qa = create_retrieval_chain(
    #     retriever=doc_search.as_retriever(), combine_docs_chain=stuff_documents_chain
    # )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # result = qa.invoke(input={"input": query})
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


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