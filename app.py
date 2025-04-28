import datetime
import json

from google.oauth2 import service_account
from dotenv import load_dotenv
from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import os
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.messages import  SystemMessage
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from chat.chat import chatWithAgent

# Load environment variables from .env
load_dotenv()
app = Flask(__name__)

# Set up CORS
CORS(app)

@app.route("/")
def hello_world():
    # chat = ChatXAI(
    # # xai_api_key="YOUR_API_KEY",
    # model="grok-beta")
    # for m in chat.stream("Who is the president of the USA?"):
    #     print(m.content, end="", flush=True)
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

    technicalPrototypePrompt = "You are senior game developer who specializes in rapid prototyping " \
    "and engine-level problem-solving.  I will describe a game mechanic or idea, and your task " \
    "is to evaluate the feasibility of a game concept by analyzing technical requirements, " \
    "identifying skill required in the design and development team, and recommend tools/engines" \
    " and your role is to evaluate its technical feasibility. Also, for every information you" \
    " send as a response to the user, if you have used any tool, add the link to the article " \
    "or sources at the end of the chat."


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

    # Add the initial system message to the chat history
    systemMessage = SystemMessage(content=technicalPrototypePrompt)
    chat_history.add_message(systemMessage)

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
        "content": formatted_result,
        "type": result.type,
        "shouldAnimate": True,
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
    # model = ChatOpenAI(model="gpt-4o")
    #chat_uid = str(uuid.uuid4())
    
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
        session_id=sessionId,
        collection="chat_history",
        client=client,
    )

    
    # Add the user message to the chat history
    chat_history.add_user_message(query)

    # Generate a response from the model
    # result = model.invoke(chat_history.messages)
    result = chatWithAgent(query,sessionId)

    formatted_result = str(result.content)
    # @stream_with_context
    # def generate_response():
    #     response_buffer = ""
    #     for chunk in chatWithAgent(query, sessionId):
    #         # Save the chunk to Firestore incrementally
    #         chat_history.add_ai_message(chunk)

    #         # Append the chunk to the response buffer
    #         response_buffer += chunk

    #         # Yield the chunk to the client
    #         yield json.dumps({"result": True,
    #     "message": "Conversation continued successfully.",
    #     "content": response_buffer,
    #     "type": "ai",})

    #     # Save the full response to Firestore
    #     chat_history.add_ai_message(response_buffer)
        
    #     return response_buffer

    # print(generate_response())
    # # response_stream = generate_response()
    # response = Response(
    #     generate_response(),
    #     content_type='application/json',
    # )

    # return response
    # Add the model's response to the chat history
    chat_history.add_ai_message(result)

    return {
        "result": True,
        "message": "Conversation continued successfully.",
        "content": formatted_result,
        "type": result.type,
        "shouldAnimate": True,
        },200
    
    


#get chat history from firestore, the payload would be a username and a collection name and a session id
@app.route("/chat_history", methods=["POST"])
def get_chat_history():
    # Get the username, collection name, and session id from the request
    data = request.get_json()
    sessionId = data.get("sessionId")

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
    if os.environ.get("ENV") == "production":
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        app.run(debug=True)

