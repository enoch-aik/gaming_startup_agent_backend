import os
import getpass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor,create_structured_chat_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory 


from memory.memory import getChatMemory


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def get_current_date_and_time(*args, **kwargs):
    """Returns the current date and time in YYYY-MM-DD H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %I:%M %p")

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."

def chatWithAgent(query,sessionId):
    load_dotenv()
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    #load the main chat model
    model = ChatOpenAI(model="gpt-4o")

    # Load the correct JSON Chat Prompt from the hub
    # prompt = hub.pull("hwchase17/structured-chat-agent")

    # Create the summarization chain
    # summary_chain = LLMChain(
    #     llm=OpenAI(),
    #     prompt=prompt,
    #     verbose=True,
    #     memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
    # )
    tools = [
        Tool(
        name="Date",
        func=get_current_date_and_time,
        description="Useful for when you need to know the current date and time.",
        ),
    # Tool(
    #     name="Wikipedia",
    #     func=search_wikipedia,
    #     description="Useful for when you need to know about a topic.",
    # ),
    # Tool(
    #     name="Tavily",
    #     func=use_tavily_search,
    #     description="Useful for when you need to search about current information on a topic",
    # ),
    TavilySearch(max_results=5,tavily_api_key="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",
                #  include_domains=["gamedeveloper.com","ign.com","youtube.com","polygon.com"],
                 ),


    ]
    prompt = hub.pull("hwchase17/structured-chat-agent")
    previousMemory = getChatMemory(sessionId)
    chat_message_history = ChatMessageHistory()
    chat_message_history.add_messages(previousMemory)

    inMemoryHistory =  InMemoryChatMessageHistory(messages=previousMemory)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=inMemoryHistory,
    )

    # Add the previous messages to the memory
    # memory.chat_memory.aadd_messages(previousMemory)
    
    # prompt = hub.pull("hwchase17/react")   

    #combining the agent with the tools and the model

    # agent = create_react_agent(model, tools, prompt=prompt)
    agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,  
        handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

    # agent_executor = AgentExecutor(agent=agent, tools=tools,memory=memory,verbose=True)

    # for step in agent.stream(
    #     {"messages": query},
    #     stream_mode="values",
    # ):
    #    step["messages"][-1].pretty_print()

    # response = agent.invoke({"messages": [("human", query)]})
    # response = agent_executor.invoke({"input": query})
    # print("Response from agent:")
    # print(response)
    # print(response)

    # agent_executor = AgentExecutor.from_agent_and_tools(
    # agent=agent,
    # tools=tools,
    # verbose=True,
    # memory=memory,  
    # handle_parsing_errors=True,  # Handle any parsing errors gracefully
    # )

    # print(memory.chat_memory.messages)

    response = agent_executor.invoke( {"input": query})

    
    return AIMessage(content=response["output"])
    # return AIMessage(content=response["messages"][-1].content)
    