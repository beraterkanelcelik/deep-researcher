import datetime
import json
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .schemas import TaskPlan


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Only supports basic arithmetic.

    Args:
        expression: A mathematical expression like '2 + 3 * 4'
    """
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression. Only basic arithmetic is supported."
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily for up-to-date information.

    Args:
        query: The search query to look up on the web.
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search = TavilySearchResults(max_results=3)
        results = search.invoke({"query": query})
        return json.dumps(results, indent=2) if isinstance(results, list) else str(results)
    except Exception as e:
        return f"Error performing web search: {e}"


@tool
def create_plan(goal: str) -> str:
    """Create a structured task plan for achieving a goal. Returns a JSON plan with ordered tasks.

    Args:
        goal: The high-level goal or objective to plan for.
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    structured_llm = llm.with_structured_output(TaskPlan)
    plan = structured_llm.invoke(
        f"Create a detailed, actionable task plan for the following goal:\n\n{goal}\n\n"
        "Break it down into concrete steps with priorities."
    )
    return plan.model_dump_json(indent=2)


@tool
def deep_research(topic: str, depth: str = "standard") -> str:
    """Launch a deep research investigation on a topic. This spawns parallel web searches,
    synthesizes findings into a structured report, and allows you to review the results.

    Args:
        topic: The research topic or question to investigate.
        depth: Research depth - 'quick' (3 searches), 'standard' (5), or 'deep' (8).
    """
    # Sentinel tool - body never executes; should_continue intercepts
    # by tool name and routes to the deep research subgraph.
    return "Routing to deep research subgraph..."


@tool
def save_report(report_index: int = 0) -> str:
    """Save a research report from the current conversation to the database.

    Args:
        report_index: Index of the report to save (0 = most recent). Defaults to 0.
    """
    # Sentinel tool - body never executes; should_continue intercepts
    # by tool name and routes to the save_confirm node.
    return "Routing to save confirmation..."


# Tools that are actually executed by ToolNode (simple tools)
SIMPLE_TOOLS = [get_current_time, calculator, tavily_search, create_plan]

# All tools bound to the LLM (includes sentinels for routing)
ALL_TOOLS = [get_current_time, calculator, tavily_search, create_plan, deep_research, save_report]

# Keep backward compat alias
TOOLS = SIMPLE_TOOLS
