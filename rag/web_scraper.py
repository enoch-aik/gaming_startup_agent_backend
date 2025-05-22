import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from urllib.parse import urlparse

# Load environment variables from .env
load_dotenv()

# Define the URL to crawl
url = "https://adventuregamers.com/articles/reviews"
parsed_url = urlparse(url)
# indexName = parsed_url.path.lstrip("/")  # Extract the path after .com/
indexName = "ai-agent-test"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")

def create_vector_store():
    # Load environment variables from .env
    load_dotenv()
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    # Defining the Firecrawl API key
    
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    # Crawl the website using FireCrawlLoader
    print("Begin crawling the website...")
    loader = FireCrawlLoader(
        api_key=api_key, url=url,  mode="crawl")
    docs = loader.load()
    print("Finished crawling the website.")

    # Convert metadata values to strings if they are lists
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", url)

        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Step 2: Split the crawled content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Step 3: Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 4: Create and persist the vector store with the embeddings
    print("\n--- Creating vector store in Pinecone")
    db = PineconeVectorStore.from_documents(
        split_docs, embeddings,index_name=indexName
    )
    return db



# Check if the Chroma vector store already exists
# if not os.path.exists(persistent_directory):
# else:
#     print(
#         f"Vector store {persistent_directory} already exists. No need to initialize.")


# db = PineconeVectorStore(index_name=indexName,
#             embedding=embeddings)

# db = create_vector_store()



def query_vector_store(query):
    """Query the vector store with the specified question."""

    # Load environment variables from .env
    load_dotenv()
    # Load the vector store with the embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a retriever for querying the vector store
    db = PineconeVectorStore(index_name=indexName,
                             embedding=embeddings)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    # Retrieve relevant documents based on the query
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Define the user's question
query = "What design concepts were discussed in this?"

# Query the vector store with the user's question
query_vector_store(query)