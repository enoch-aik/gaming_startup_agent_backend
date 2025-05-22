import os
import json

from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from google.oauth2 import service_account


def getChatMemory(session_name):
    """
    Get the chat memory for a given session name.
    Args:
        session_name (str): The name of the session.
    Returns:
        list: A list of chat messages.
    """
    # Load environment variables from .env file
    # Check if the app is in deployment or production
    if os.environ.get("ENV") == "production":
        # Use credentials from the environment variable
        credentials_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
    else:
        # Exclude credentials in deployment
        credentials = None

    # Initialize Firestore Client
    client = firestore.Client(project="game-startup-ai-agent", credentials=credentials)

    # Initialize Firestore Chat Message History
    chat_history = FirestoreChatMessageHistory(
        session_id=session_name,
        collection="chat_history",
        client=client,
    )


    # Get the chat history
    return chat_history.messages
