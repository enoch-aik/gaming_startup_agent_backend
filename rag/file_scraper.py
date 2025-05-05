import os
import markdown


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup

currentDir = os.path.dirname(os.path.abspath(__file__))
scrapedDataDir = os.path.join(currentDir, "scraped_data")
indexName = "ai-agent-test"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_all_files_in_scraped_data():
    """Returns a list of all valid text files in the /scraped_data directory."""
    if not os.path.exists(scrapedDataDir):
        raise FileNotFoundError(f"The directory {scrapedDataDir} does not exist.")
    
    # List all valid text files in the directory
    valid_extensions = {".txt", ".md"}  # Add other extensions if needed
    files = []
    for root, _, filenames in os.walk(scrapedDataDir):
        for filename in filenames:
            if os.path.splitext(filename)[1] in valid_extensions:
                files.append(os.path.join(root, filename))
    
    return files





def convert_filename_to_url(filename):
    """
    Converts a file name into the desired URL format.
    
    Example:
    Input: www_gamedeveloper_com_programming_get_a_job_join_jackbox_games_as_a_sr_gameplay_engineer_2.md
    Output: www.gamedeveloper.com/programming-get-a-job-join-jackbox-games-as-a-sr-gameplay-engineer-2
    """
    # Remove the file extension
    if filename.endswith(".md"):
        filename = filename[:-3]
    
    # Replace the first underscores (before 'com') with dots
    parts = filename.split("_com_", 1)
    if len(parts) == 2:
        domain_part = parts[0].replace("_", ".")
        path_part = parts[1]
        # Replace the first underscore after 'com' with a slash
        path_part = path_part.replace("_", "-", path_part.count("_"))
        return f"https://{domain_part}.com/{path_part}"
    else:
        return filename



# def create_vector_store():
    
#     files = get_all_files_in_scraped_data()
#     documents = []
    

#     for file in files:
#         loader = TextLoader(file)
#         book_docs = loader.load()
#         for doc in book_docs:
#             #print the file that has been selected
#             print(f"Processing file: {file}")
#             # Add metadata to each document indicating its source
#             doc.metadata = {"source": convert_filename_to_url(os.path.basename(file))}
#             documents.append(doc)
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=70)
#             docs = text_splitter.split_documents(documents)
#             # break down the docs into smaller chunks and add to vector store
            
            
#             # Helper function to split docs into smaller batches
#             def split_into_batches(docs, batch_size):
#                 for i in range(0, len(docs), batch_size):
#                     yield docs[i:i + batch_size]

#             # Process each batch of documents
#             batch_size = 100  # Adjust the batch size as needed to stay within the 2MB limit
#             for batch in split_into_batches(docs, batch_size):
#                 PineconeVectorStore.from_documents(batch, embeddings, index_name=indexName)


def clean_markdown_with_markdown(content):
    # Convert Markdown to HTML
    html = markdown.markdown(content)
    
    # Use BeautifulSoup to extract plain text from HTML
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def create_vector_store():
    files = get_all_files_in_scraped_data()

    for file in files:
        loader = TextLoader(file)
        book_docs = loader.load()
        documents = []  # Reset the documents list for each file

        for doc in book_docs:
            # Print the file being processed
            print(f"Processing file: {file}")

            doc.page_content = clean_markdown_with_markdown(doc.page_content)

            # Add metadata to each document indicating its source
            doc.metadata = {"source": convert_filename_to_url(os.path.basename(file))}
            documents.append(doc)

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Helper function to split docs into smaller batches
        def split_into_batches(docs, batch_size):
            for i in range(0, len(docs), batch_size):
                yield docs[i:i + batch_size]

        # Process each batch of documents
        batch_size = 100  # Adjust the batch size as needed to stay within the 2MB limit
        for batch in split_into_batches(docs, batch_size):
            PineconeVectorStore.from_documents(batch, embeddings, index_name=indexName)            

create_vector_store()


def query_vector_store(query):
    """Query the vector store with the specified question."""
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

# # Define the user's question
# query = "What design concepts were discussed in this?"


# # Query the vector store with the user's question
# query_vector_store(query)

    










