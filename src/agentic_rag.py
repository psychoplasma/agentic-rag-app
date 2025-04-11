import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from typing import Any, Literal

from pydantic import BaseModel, Field

from langchain_google_vertexai import VertexAIEmbeddings

from src.document_processors.javacript_code_processor import JSCodeDocumentProcessor
from src.vector_store.chromadb import ChromaDB
from src.vector_store.vertexai_vector_search import VertexAIVectorStore
from src.tools.javascript_executor.tool import JSCodeExecutor


PY_ENV = os.environ.get("PY_ENV")
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("INDEX_REGION")
BUCKET_URI = os.environ.get("BUCKET_URI")
INDEX_ID = os.environ.get("INDEX_ID")
INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID")

document_processor = JSCodeDocumentProcessor()
llm = init_chat_model(
    "gemini-2.0-flash-001",
    model_provider="google_vertexai",
)

vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    bucket_uri=BUCKET_URI,
    index_id=INDEX_ID,
    index_endpoint_id=INDEX_ENDPOINT_ID,
    embeddings=VertexAIEmbeddings(model="text-embedding-005"),
) if PY_ENV == "prod" else ChromaDB(
    embeddings=VertexAIEmbeddings(model="text-embedding-005"),
    collection_name="js_code_collection",
)

RECURSION_LIMIT = 5

@tool(response_format="content_and_artifact")
def retriever(query: str):
    """Retrieves javascript code snippets related to the given query"""
    print("Executing retriever...")

    retrieved_docs = vector_store.search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

def grade_documents(state: MessagesState) -> Literal["generate", "rewrite"]:
    """Determines whether the retrieved documents are relevant to the question."""

    # Data model
    class Grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm_with_structured_output = llm.with_structured_output(Grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Retrieved document: \n\n {context} \n\n
        User question: \n\n {question} \n\n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    grade = (prompt | llm_with_structured_output).invoke({
        "question": state["messages"][0].content,
        "context": state["messages"][-1].content,
    })

    return "generate"if grade.binary_score == "yes" else "rewrite"

def agent(state: MessagesState):
    """Generate tool call for retrieval or respond."""

    llm_with_tools = llm.bind_tools([retriever])
    response = llm_with_tools.invoke(state["messages"][0].content)

    tool_calls =[
        {
            "name": "retriever",
            "args": {"query": state["messages"][0].content},
            "id": "tool_call_id_1234abced",
            "type": "tool_call",
        }
    ]
    response.tool_calls = tool_calls

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def generate(state: MessagesState):
    msg = HumanMessage(
        content=f""" \n 
        You are an assistant for code generation tasks for user questions.\n\n
        Here is the user question:\n{state["messages"][0].content}\n\n
        Here is the context:\n{state["messages"][-1].content}\n\n

        Generate a code snippet in javascript which includes a function to achieve the task described in the user question based on the context.
        If you don't know how to generate code for the task, say that you don't know. 
        If you have a generated code, do not explain the code neither give examples 
        nor suggestions to imporve it. And do not use markdown formatting.
        And resulting code should always end with call to console.log function which takes the output of the call to the previous function as an argument.
        For example: if the generated function to achieve user's task is as follows;\n
        function foo(a, b) '{' return a + b; '}'\n
        Then the resulting code should be like this:\n
        function foo(a, b) '{' return a + b; '}'
        console.log(foo(1, 2));
        \n\n
        """,
    )
    
    response = llm.invoke([msg])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def rewrite(state: MessagesState):
    """Transform the query to produce a better question."""

    msg = [
        HumanMessage(
            content=f""" \n 
            Look at the input and try to reason about the underlying semantic intent / meaning.\n 
            Here is the initial question:\n\n{state["messages"][0].content}\n\n
            Formulate an improved question: """,
        )
    ]

    response = llm.invoke(msg)
    return {"messages": [response]}

def generate_with_conversation(state: MessagesState):
    """Generate answer."""

    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    prev_context = "\n\n".join(doc.content for doc in tool_messages)
    system_message = SystemMessage(
        content=f""" \n 
        You are an assistant for code generation tasks for user questions.\n\n
        Here is the user question:\n{state["messages"][0].content}\n\n
        Here is the context:\n{prev_context}\n\n

        Generate a code snippet in javascript which includes a function to achieve the task described in the user question based on the context.
        If you don't know how to generate code for the task, say that you don't know. 
        If you have a generated code, do not explain the code neither give examples 
        nor suggestions to imporve it. And do not wrap your answer in a code block which is like ```javascript ... ```.
        Resulting code should always be calling in the function with arguments taken from the user question.
        And resulting code should always ends with console.log function call which takes the output of the previous function call as an argument.
        For example: if the generated function to achieve user's task as follows;\n
        function foo(a, b) '{' return a + b; '}'\n
        Then the resulting code should be like this:\n
        function foo(a, b) '{' return a + b; '}'
        console.log(foo(1, 2));
        \n\n
        """,
    )

    # Messages except the tool messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # Feed llm with the combined messages from memory
    response = llm.invoke([system_message] + conversation_messages)

    return {"messages": [response]}

def execute(state: MessagesState):
    """Execute generated code."""

    code = state["messages"][-1].content
    cleaned_code = code.replace("```javascript", "").replace("```", "").strip()

    try:
        result = JSCodeExecutor.execute(cleaned_code)
        return {"messages": [result]}
    except Exception as e:
        error_message = f"Error executing code: {str(e)}"
        print(error_message)
        return {"messages": [HumanMessage(content=error_message)]}

def build_agent():
    """Compile langgraph workflow."""

    # memory = MemorySaver()
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("execute", execute)
    workflow.add_node("grade_documents", grade_documents)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {END: "grade_documents", "tools": "retrieve"},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", END)

    return workflow.compile()

    # return workflow.compile(checkpointer=memory)

def ask_agent(
    agent: CompiledStateGraph,
    query: str,
    thread_id: str,
    user_id: str,
) -> (dict[str, Any] | Any):
    # TODO: Check whether conversation history is empty,
    # and fetch conversation history if it is so.

    steps = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config= {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        },
    )

    res = []
    for step in steps:
        step["messages"][-1].pretty_print()
        # Get the last message from the final state
        res = step

    # res = agent.invoke(
    #     {"messages": [{"role": "user", "content": query}]},
    #     config= {"configurable": {"thread_id": thread_id}},
    # )

    # TODO: save conversation history to database

    return {"answer": res["messages"][-1].content, "context": []}

def process_repository(path: str) -> int:
    docs = document_processor.process(path)
    return vector_store.add(docs)
