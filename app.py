import datetime

from dotenv import load_dotenv
from flask import Flask, request
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/start_chat", methods=["POST"])
def start_chat():
    # Get the username and collection name from the payload

    data = request.get_json()
    username = data.get("username")
    agentType = data.get("agentType")
    query = data.get("query")

    if not username:
        return {"result":False,
            "error": "Username is required."}, 400
    
    if not agentType:
        return {"result":False,
            "error": "Agent Type is required."}, 400
    
    if not query:
        return {"result":False,
            "error": "Query is required to start conversation."}, 400

    technicalPrototypePrompt = "You are senior game developer who specializes in rapid prototyping and engine-level problem-solving.  I will describe a game mechanic or idea, and your task is to evaluate the feasibility of a game concept by analyzing technical requirements, identifying skill required in the design and development team, and recommend tools/engines and your role is to evaluate its technical feasibility"


    #load the small 3.5 chat model  to only summarize the first query
    querySummarizerModel = ChatOpenAI(model="gpt-3.5-turbo")

    
    #load the main chat model
    model = ChatOpenAI(model="gpt-4o")
    #chat_uid = str(uuid.uuid4())
    

    #invoke small model to summarize the first query
    formattedQuery = querySummarizerModel.invoke("I want you to summarize the following " \
    "query into 3 to 4 words in such a way that it is a 2nd party summary: " + query)

    currentTimestamp = datetime.datetime.now().timestamp()
  
    session_name = username + "_"+formattedQuery.content + "_" + str(currentTimestamp)
    
    # Initialize Firestore Client
    client = firestore.Client(project="game-startup-ai-agent")

    # Initialize Firestore Chat Message History
    chat_history = FirestoreChatMessageHistory(
        session_id=session_name,
        collection="chat_history",
        client=client,
    )

    # Add the initial system message to the chat history
    chat_history.add_user_message(technicalPrototypePrompt)

    # Add the user message to the chat history
    chat_history.add_user_message(query)

    # Generate a response from the model
    result = model.invoke(chat_history.messages)

    formatted_result = str(result.content)
    # Add the model's response to the chat history
    chat_history.add_ai_message(formatted_result)

    return {
        "result": True,
        "message": "Chat started successfully.",
        "title":formattedQuery.content,
        "sessionId": session_name,
        "data": formatted_result,
        "type": result.type,
        }, 200


@app.route("/continue_chat", methods=["POST"])
def continue_chat():
    # Get the username and collection name from the payload

    data = request.get_json()
    sessionId= data.get("sessionId")
    query = data.get("query")

    if not query:
        return {"result":False,
            "error": "Query is required to continue chat."}, 400
    
    if not sessionId:
        return {"result":False,
            "error": "SessionID is required to continue chat."}, 400
    
    #load the main chat model
    model = ChatOpenAI(model="gpt-4o")
    #chat_uid = str(uuid.uuid4())
    
    # Initialize Firestore Client
    client = firestore.Client(project="game-startup-ai-agent")

    # Initialize Firestore Chat Message History
    chat_history = FirestoreChatMessageHistory(
        session_id=sessionId,
        collection="chat_history",
        client=client,
    )

    
    # Add the user message to the chat history
    chat_history.add_user_message(query)

    # Generate a response from the model
    result = model.invoke(chat_history.messages)

    formatted_result = str(result.content)
    # Add the model's response to the chat history
    chat_history.add_ai_message(formatted_result)

    return {
        "result": True,
        "message": "Conversation continued successfully.",
        "content": formatted_result,
        "type": result.type,
        }, 200


#get chat history from firestore, the payload would be a username and a collection name and a session id
@app.route("/chat_history", methods=["POST"])
def get_chat_history():
    # Get the username, collection name, and session id from the request
    data = request.get_json()
    sessionId = data.get("sessionId")

    client = firestore.Client(project="game-startup-ai-agent")

    # Initialize Firestore Chat Message History
    chat_history = FirestoreChatMessageHistory(
        session_id=sessionId,
        collection="chat_history",
        client=client,
    )
    messages = chat_history.messages

    # Convert messages to a list of dictionaries
    messages = [
        {
            "type": message.type,
            "content": message.content if hasattr(message, 'content') else str(message),
        }
        for message in messages
    ]


    return {
        "result": True,
        "message": "Chat history retrieved successfully.",
        "data": messages}, 200

    

if __name__ == "__main__":
    app.run(debug=True)


