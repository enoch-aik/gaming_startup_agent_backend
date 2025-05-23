import os
import getpass
import sys

# Add the custom langchain-tavily dependency as a directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../tools/langchain-tavily"))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import (AgentExecutor, create_react_agent)
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import YouTubeSearchTool

from memory.memory import getChatMemory
from rag.vector_store import customRetriever, contextualize_q_prompt, qa_prompt
from tools.image_gen.image_gen import generateDalle3Image, editDalle3Image, edit_tool_parser
from tools.file_storage.file_storage import store_file
from tools.sound_effect.sound_effect import generate_sound_effect


# 0.3.23
# function used to get the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


# function used to get the current date and time
def get_current_date_and_time(*args, **kwargs):
    """Returns the current date and time in YYYY-MM-DD H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %I:%M %p")

# function used to search wikipedia
def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except Exception:
        return "I couldn't find any information on that."


# this is the primary function that is used to chat with the agent, it consists of the query and sessionId
def chatWithAgent(query, sessionId):
    """
    Chat with the agent using the given query and session ID.
    
    Args:
        query (str): The user's query.
        sessionId (str): The session ID for the chat.
    """
    # index name of the vector store created on Pinecone
    indexName = "ai-agent-test"
    # load the environment variables from the .env file
    load_dotenv()
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    
    # initialize the model and embedding
    model = ChatOpenAI(model="gpt-4o")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    # prompt = hub.pull("hwchase17/structured-chat-agent")
    # react_docstore_prompt = hub.pull("hwchase17/react")
    # The tag ![Music] has to be used in the out so that the user can preview the sound effect in the markdown.
    # If you need to use the Generate sound effect tool, you should give the output in a markdown format ![Music](https://storedfile.mp3) .
    
    # This is the prompt that would be used throughout the conversation with the agent.
    # It included the instructions for the agent to follow and also the tools that are available to the agent.
    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history", "technicalOrDigitalPrototypingPrompt"],
        partial_variables={"format_instructions": edit_tool_parser()},
        template="""
        You are a great AI-Agent that is designed to assist game developers and game startups
        or studios to test their ideas and validate their assumptions. 
        Your default language is English and you have access to additional 
        tools in order to answer the following questions as best you can. Always answer in the same language as 
        the user question. You have access to the following tools:

        {tools}

        You should also access the chat history to answer the question. You can use the tools to get more information,
         but you should only use them if you think it will help you answer the question better.
        
        Always prioritize the RAG tool and TavilySearch tool as it uses Retrieval-Augmented Generation (RAG)
         to answer the question and after using this tool, check if you need to use any other tool. After that, you 
         return the answer to the user.
        The chat history is as follows:
        {chat_history}

        If you need to use the Edit image tool, you should only use it once and you must include both the "image" 
        and "query" fields in the Action Input. an exampe is below
        Action Input: {{"image":"https://storage.googleapis.com/game-startup-ai-agent.firebasestorage.app/img-gsHV1XsWihjjMErL3xpl06Cm.png",
        "query":"Add a cat to the image"}}.

        The Edit image tool would return an image url that has already been stored on FirebaseStorage, so you 
        return the image to the user alongise a description of the image and the thought process behind the 
        design and in a case where the user is modifying the image, add futher description based on the user's request.

        If you need to use the Store file tool, make sure that there are no extra spaces or characters like 
        ``` or " added to the image url

        If you need to use the Generate image tool, once the image is created, use the Store file tool to 
        store the image in Firebase Storage and the Store File tool would return the download URL. 
        Your response should have the stored image url from FirebaseStorage first (do not send the result 
        of the Generate image tool as a response). After, you give a description of the image and the thought
         process behind the design and in a case where the user is modifying the image, add futher description
          based on the user's request.
        
        Always try to see if you can use the tools and for responses, include the link to the article or 
        sources at the end of the chat. You can use the tools to get more information, but you should only
         use them if you think it will help you answer the question better.
        
        If you are asked a question and you try to use a tool but it fails, you should try to answer the 
        question without using the tool. If you are not able to answer the question, you should say that
         you are not able to answer the question and suggest that the user try again later.

        If you are asked about any information that requires getting the latest information, it should be dated to 2025
        
        Also, if you are asked about current statistics for a game try to search these websites 
        https://games-stats.com/steam/tags/ , https://steamdb.info

        If you are asked a question about a game idea, you should try to answer the 
        question by comprehensively explaining about how feasible the game is in terms of
         development and also skills needed to develop the game, for each skill mention the 
         number of people that would be needed, and an estimate of the time to develop and publish such a game. 
         Also identify the potential genre and the current 
         market or userbase of the game, remember to use the Answer Question tool or Tavily 
         Search tool if you have not used it aready to get more information on questions about 
         game ideas. Once you are done, return the answer comprehensively and also include a 
         link to the article or sources at the end of the chat. 
        To use a tool, please use the following format:

        
        The for every image you return to the user, it should be in the format 
        ![Image](< put the image url here so that it can be decoded in the markdown).
        '''
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat 2 times)
        '''

        When you have a response to respond comprehensively to the Human, or if 
        you do not need to use a tool, you MUST use the format:
        '''
        Thought: Do I need to use a tool? No
        '''
        Final Answer: [your response here]
        

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
    )
    # get the chat history from the firebase storage and add it to the chat message history
    previousMemory = getChatMemory(sessionId)
    chat_message_history = ChatMessageHistory()
    chat_message_history.add_messages(previousMemory)

    inMemoryHistory = InMemoryChatMessageHistory(messages=previousMemory)

    # Memory buffer to main context of the conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=inMemoryHistory,
    )
    # custom retriever to retrieve the top similar results from Pinecone using Maximal Marginal Relevance (MMR)
    # The top 10 similar results are retrieved and then the top 5 are selected based on MMR with lambda_mult = 0.5
    retriever = customRetriever(
        indexName,
        query,
        embedding,
        "mmr",
        {"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
    )

    # this is the system prompt that would be used to contextualize the question (it has been initialzed in the vector_store.py file)
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    # this is the system prompt that would be used to answer the question (it has been initialzed in the vector_store.py file)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # this is the RAG chain that combines the history aware retriever and the question answer chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # this is the list of tools that would be used by the agent to answer the question
    # each tool was described to allow the agent to understand what it does and how to use it
    tools = [
        TavilySearch(
            max_results=5,
            description="Useful for when you need to answer questions about the latest news",
            #  include_domains=["gamedeveloper.com","ign.com","youtube.com","polygon.com"],
        ),
        TavilySearch(
            max_results=5,
            description="Useful for when you need to get the latest information on the current market stats for a game genre",
            include_domains=["https://games-stats.com/steam/tags/"],
            #  include_domains=["gamedeveloper.com","ign.com","youtube.com","polygon.com"],
        ),
        TavilySearch(
            max_results=5,
            description="Useful for when you need to get the latest information on the statistics for a game",
            include_domains=["https://steamdb.info"],
        ),
        Tool(
            name="RAG tool",
            func=lambda input, **kwargs: rag_chain.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", memory.chat_memory.messages)}
            ),
            description="Useful when you need to answer questions using RAG",
        ),
        YouTubeSearchTool(),
        Tool(
            name="Generate image",
            func=generateDalle3Image,
            description="Useful for when you need to generate images from text",
        ),
        # editDalle3Image,
        Tool(
            name="Generate sound effect",
            func=generate_sound_effect,
            description="Useful for when you need to generate sound effects from text",
        ),
        Tool(
            name="Edit image",
            func=editDalle3Image,
            description="Useful for when you need to edit images when a previous image has been generated and the user wants to modify it",
        ),
      
        # ),
        # StructuredTool.from_function(
        #     editDalle3Image,
        #     name="Edit image",
        #     description="Useful for when you need to edit images when a previous image has been generated and the user wants to modify it",
        #     args_schema=EditImageStringsArgs,
        # ),
        # Tool(
        #     name="Get image file",
        #     func=get_image_file_from_url,
        #     description="Useful for when you need to get the image file from a URL"
        # ),
        Tool(
            name="Store file",
            func=store_file,
            description="Useful for when you need to store files in Firebase Storage after generating them",
        ),
    ]

    # this is the ReAct (Reasoning and Acting) agent that would autonomously use the tools available to the query of the user
    agent = create_react_agent(
        model,
        tools,
        prompt
    )

    # this is the agent executor that would execute the agent and return the response
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    response = agent_executor.invoke(
        {"input": query, "chat_history": memory.chat_memory.messages}
    )

    return AIMessage(content=response["output"])

    # this step is for streaming the response from the agent, but this was not used since different tasks 
    # like saving response and formatting response are used in the endpoints
    # for step in agent_executor.stream({"input": query}):
    #     yield step["messages"][-1].content
