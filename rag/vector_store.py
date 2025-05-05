from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()


# Function to query a vector store with different search types and parameters
def queryVectorStore(indexName, query, embedding, search_type, search_kwargs):
    db = PineconeVectorStore(index_name=indexName, embedding=embedding)
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    relevant_docs = retriever.invoke(query)
    # Display the relevant results with metadata
    return relevant_docs


def customRetriever(indexName, query, embedding, search_type, search_kwargs):
    db = PineconeVectorStore(index_name=indexName, embedding=embedding)
    return db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone comprehensive question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history


# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are senior game developer who specializes in rapid prototyping and engine-level problem-solving, your task is to comprehensively evaluate the feasibility of a game concept by analyzing technical requirements, identifying skill required in the design and development team, and recommend tools/engines and your role is to evaluate its technical feasibility. You should also look into identifying the potential genre and the current market or userbase of the game. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Do not try to make up an answer. "
    "Try to always give your output in json format. "
    "Try to give the answer in only one sentence. "
    "Always include the source of the information in your answer. "
    "If the question is not related to the context, say that "
    "it is not related and ask the user to rephrase or provide "
    "more context. "
    "If the question is too vague, ask the user to clarify. "
    "Look to give the user comprehensive answers."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
