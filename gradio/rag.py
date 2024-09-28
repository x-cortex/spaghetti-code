import os
import getpass
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
import tempfile
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List, Annotated
import operator

from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


# Create a table if it doesn't exist
def initialize_database():
    # Supabase does not require explicit table creation in the same way as SQLite
    print("Supabase client initialized")


initialize_database()


# Initialize LLMs
local_llm = "llama3.2:3b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Set environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = os.getenv(var)


_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Initialize tools
web_search_tool = TavilySearchResults(k=3)

# Setup Vector Store
persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
raw = TextLoader("processed_chat.txt").load()


class Every20LinesSplitter(TextSplitter):
    def split_text(self, text: str) -> list[str]:
        lines = text.split("\n")
        return ["\n".join(lines[i : i + 10]) for i in range(0, len(lines), 10)]


text_splitter = Every20LinesSplitter(chunk_overlap=0)
docs = text_splitter.create_documents([raw[0].page_content])

vecstore = SKLearnVectorStore.from_documents(
    docs[:20],
    NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    persist_path=persist_path,
)

vecstore.persist()
print("Vector store was persisted to", persist_path)

# Create retriever
retriever = vecstore.as_retriever(k=3)


# Define Graph State
class GraphState(TypedDict):
    chat_message: str  # User question
    route: str  # Route to use for answer generation
    documents: List[str]  # List of retrieved documents from vector store or web search
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]


# Prompt Instructions
router_instructions = """You are an expert at routing a user question to a vectorstore or quick-response.
The vectorstore contains documents related to chat related to various general topic with more depth and more tokens.                    
Use the vectorstore for questions on these topics. For all else, and especially small and generic topics.
Return JSON with single key, datasource, that is 'quick-response' or 'vectorstore' depending on the question.
"""

doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user chat message.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question.
"""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here are few of the user chat message \n\n {chat_message}
Observe this carefully and give a binary score to indicate whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question.
"""

rag_prompt = """You are responding as a person to a chat response. Here is the context to use to answer the question:
{context} 
Think about this context carefully and give an appropriate response to the following set of chat messages: {chat_message}
Provide reply in a conversational manner using only the context provided. Use one sentence maximum.
"""

quick_response_prompt = """You are responding to a person's chat message.
Here is the chat message: {chat_message}
Its a generic chat message. So keep the response short and concise. Don't use more than 3 words.
"""


# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Graph Nodes
def route_question(state):
    """Route question to web search or RAG"""
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["chat_message"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "quick-response":
        return {"route": "quick-response"}
    elif source == "vectorstore":
        print("Routing question to vector store")
        return {"route": "vectorstore"}


def route_question_holder(state):
    return state["route"]


def retrieve(state):
    """Retrieve documents from vector store"""
    print("Retrieving documents from vector store")
    chat_message = state["chat_message"]
    documents = retriever.invoke(chat_message)
    return {"documents": documents, "route": state["route"]}


def generate(state):
    """Generate answer using RAG on retrieved documents"""
    chat_message = state["chat_message"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(
        context=docs_txt, chat_message=chat_message
    )
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {
        "generation": generation,
        "loop_step": loop_step + 1,
        "route": state["route"],
    }


def quick_response(state):
    """Generate answer using quick-response"""
    chat_message = state["chat_message"]
    loop_step = state.get("loop_step", 0)

    quick_response_formatted = quick_response_prompt.format(chat_message=chat_message)
    generation = llm.invoke([HumanMessage(content=quick_response_formatted)])
    return {
        "generation": generation,
        "loop_step": loop_step + 1,
        "route": state["route"],
    }


# Define and compile the graph
def create_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("quick-response", quick_response)
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Build graph
    workflow.set_entry_point("route_question")
    workflow.add_conditional_edges(
        "route_question",
        route_question_holder,
        {
            "quick-response": "quick-response",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "generate")

    graph = workflow.compile()
    return graph


# Function to get response from the graph
def get_response(chat_message: str, recipient: str, chat_history: str, timestamp: str):
    g = create_graph()
    inputs = {"chat_message": chat_message}
    events = []

    for event in g.stream(inputs, stream_mode="values"):
        events.append(event)

    if events:

        last_event = events[-1]
        print(last_event)
        generation = last_event.get("generation", "No response generated.")
        print("Connecting to Supabase")
        data = {
            "recipient": recipient,
            "last_chat": chat_message,
            "chat_history": chat_history,
            "timestamp": timestamp,
            "route": events[-1]["route"],
            "response": (
                generation.content if hasattr(generation, "content") else generation
            ),
        }
        supabase.table("chat_data").insert(data).execute()
        return generation.content if hasattr(generation, "content") else generation
    return "No response generated."
