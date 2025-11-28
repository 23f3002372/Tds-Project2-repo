from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 180000


# STATE
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
]


# LLM INIT 
rate_limiter = InMemoryRateLimiter(
    requests_per_second=7 / 60,
    check_every_n_seconds=1,
    max_bucket_size=7
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)



# SYSTEM PROMPT 

SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain.
6. When the server response has "url": null or no new URL, output exactly: END

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool that's provided
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server response.
- Never stop early.
- Use tools for HTML, downloading, rendering, OCR, or running code.
- When you get a response with url: null, that means the quiz is complete - output END immediately.
- Include:
    email = {EMAIL}
    secret = {SECRET}
"""


# AGENT NODE
def agent_node(state: AgentState):
    # time-handling
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time[cur_url]
    offset = os.getenv("offset")
    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time

        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 90):
            print("Timeout exceeded — instructing LLM to submit wrong answer.", diff, "Offset=", offset)

            fail_instruction = """
            You have exceeded the time limit for this task (over 180 seconds).
            You MUST immediately output: END
            Do not call any tools. Do not try to submit anything else.
            Just respond with: END
            """

            result = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fail_instruction}
            ])
            return {"messages": [result]}

    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm,  
    )
    
    result = llm.invoke(trimmed_messages)

    return {"messages": [result]}

# ROUTE LOGIC 

def route(state):
    last = state["messages"][-1]

    # Check for tool calls first
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Route → tools")
        return "tools"

    # Check for END signal in content
    content = getattr(last, "content", None)
    
    # Handle string content
    if isinstance(content, str):
        if content.strip().upper() == "END":
            print("Route → END (detected END signal)")
            return END
    
    # Handle list content (block content)
    if isinstance(content, list) and len(content) > 0:
        first_item = content[0]
        if isinstance(first_item, dict):
            text = first_item.get("text", "").strip().upper()
            if text == "END":
                print("Route → END (detected END signal)")
                return END

    print("Route → agent")
    return "agent"



# GRAPH
graph = StateGraph(AgentState)

graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)
robust_retry = {
    "initial_interval": 1,
    "backoff_factor": 2,
    "max_interval": 60,
    "max_attempts": 10
}

graph.add_node("agent", agent_node, retry=robust_retry)
app = graph.compile()




# RUNNER

def run_agent(url: str):
    # system message is seeded ONCE here
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("Tasks completed successfully!")
