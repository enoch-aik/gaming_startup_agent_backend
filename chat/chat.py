import os
import getpass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
    create_react_agent,
)
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.openai_dalle_image_generation import (
    OpenAIDALLEImageGenerationTool,
)
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper


from memory.memory import getChatMemory
from rag.vector_store import customRetriever, contextualize_q_prompt, qa_prompt


# 0.3.23
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


def chatWithAgent(query, sessionId):
    indexName = "ai-agent-test"
    load_dotenv()
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    model = ChatOpenAI(model="gpt-4o")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # prompt = hub.pull("hwchase17/structured-chat-agent")

    # react_docstore_prompt = hub.pull("hwchase17/react")
    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
        template="""
        You are a great AI-Assistant that has access to additional tools in order to answer the following questions as best you can. Always answer in the same language as the user question. You have access to the following tools:

        {tools}

        You should also access the chat history to answer the question. You can use the tools to get more information, but you should only use them if you think it will help you answer the question better.
        
        The chat history is as follows:
        {chat_history}

        Always try to see if you can use the tools and for responses, include the link to the article or sources at the end of the chat. You can use the tools to get more information, but you should only use them if you think it will help you answer the question better.
        
        To use a tool, please use the following format:

        '''
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat 3 times)
        '''

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        '''
        Thought: Do I need to use a tool? No
        '''
        Final Answer: [your response here]
        

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
    )
    previousMemory = getChatMemory(sessionId)
    chat_message_history = ChatMessageHistory()
    chat_message_history.add_messages(previousMemory)

    inMemoryHistory = InMemoryChatMessageHistory(messages=previousMemory)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=inMemoryHistory,
    )

    retriever = customRetriever(
        indexName,
        query,
        embedding,
        "mmr",
        {"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
    )
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    api_wrapper = DallEAPIWrapper()

    tools = [
        TavilySearch(
            max_results=5,
            tavily_api_key="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",
            descrption="Useful for when you need to answer questions about the latest news",
            #  include_domains=["gamedeveloper.com","ign.com","youtube.com","polygon.com"],
        ),
        Tool(
            name="Answer Question",
            func=lambda input, **kwargs: rag_chain.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", memory.chat_memory.messages)}
            ),
            description="Useful when you need to answer questions based on a context",
        ),
        YouTubeSearchTool(),
        # OpenAIDALLEImageGenerationTool(api_wrapper=api_wrapper)
    ]

    # agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)
    agent = create_react_agent(
        model,
        tools,
        prompt
    )
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

    # for step in agent_executor.stream({"input": query}):
    #     yield step["messages"][-1].content
