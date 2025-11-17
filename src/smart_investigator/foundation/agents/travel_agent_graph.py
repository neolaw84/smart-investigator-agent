from uuid import uuid4

import random

from typing import Literal, Optional, Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from smart_investigator.foundation.agents.ticketing_agent_graph import create_ticketing_agent
from smart_investigator.foundation.agents.travel_agent_tool_agents import create_ticketing_agent_tool

@tool("get_weather", description="Get the current weather in a given location as string.")
def get_weather(location):
    weather = random.choice(["sunny", "rainy"])
    return f"The weather in {location} is {weather}."

@tool("ask_for_help", description="Ask for help or more information from the caller agent (usually human).")
def ask_for_help(question):
    help = interrupt(f"I have a question : {question}")
    return help 

class TravelAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    loop_counter: int
    active_agent: str


def create_travel_agent(llm: BaseChatModel, checkpointer: BaseCheckpointSaver, agent_tools: Optional[list]=[]):
    # this should loop through the available ones 

    tools = [get_weather, ask_for_help] + agent_tools
    tool_name_to_executables = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    def llm_call(state: TravelAgentState) -> TravelAgentState:

        _input = [SystemMessage("You are a travel agent. You can help the caller check weather information. You can also recommend hotels. Finally, you can refer the user to ticketing agent.")] + \
                 state["messages"]

        ai_message = llm_with_tools.invoke(
                input=_input
            )
        state["messages"] = [ai_message]
        state["loop_counter"] = 1 + state.get("loop_counter", 0)
        return state

    def check_tool_call(state: TravelAgentState) -> Literal["tool_executor", "END"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_executor"
        else:
            return END

    def tool_executor(state: TravelAgentState):
        result = []
        for tc in state["messages"][-1].tool_calls:
            _id, _name, _args = tc["id"], tc["name"], tc["args"]
            tool_response = tool_name_to_executables[_name].invoke(_args)
            result.append(ToolMessage(content=tool_response, tool_call_id=_id))
        state["messages"].extend(result)
        return state

    builder = StateGraph(TravelAgentState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_executor", tool_executor)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", check_tool_call, ["tool_executor", END])
    builder.add_edge("tool_executor", "llm_call")

    return builder.compile(checkpointer=checkpointer)
