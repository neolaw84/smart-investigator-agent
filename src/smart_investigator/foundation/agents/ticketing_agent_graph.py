from uuid import uuid4

import random

from typing import Literal, Optional, Annotated
from typing_extensions import TypedDict

from langchain.tools import tool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage

from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

@tool("ask_for_help_tkt", description="Ask for help from the caller agent (usually travel agent).")
def ask_for_help_tkt(question):
    help = interrupt(f"I have a question : {question}")
    return help 

@tool("calculate_total", description="Calculate the total amount from quantity and unit price.")
def calculate_total(quantity: int, unit_price: float) -> float:
    return quantity * unit_price

@tool("check_ticket_price", description="Check the price of a ticket for a given origin, destination and whether it is round-trip or one-way.")
def check_ticket_price(origin: str, destination: str, round_trip: bool) -> float:
    base_price = 100.0
    if round_trip:
        return base_price * 2
    return base_price

class TicketingAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    loop_counter: int

def create_ticketing_agent(llm: BaseChatModel, checkpointer: BaseCheckpointSaver, agent_tools: Optional[list]=None):
    MAX_LOOPS = 3
    tools = [ask_for_help_tkt, calculate_total, check_ticket_price]
    tool_name_to_executables = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    def llm_call(state: TicketingAgentState) -> TicketingAgentState:
        if state.get("loop_counter", 0) > MAX_LOOPS:
            state["messages"] = [AIMessage(content="It seems I am going around. Goodbye!")]
            return state

        _input = [SystemMessage("You are a ticketing agent. \
                                You can help calculate the total ticket price. And print invoice."
                            )] + \
                 state["messages"]

        ai_message = llm_with_tools.invoke(
                input=_input
            )
        state["messages"] = [ai_message]
        state["loop_counter"] = 1 + state.get("loop_counter", 0)
        return state

    def check_tool_call(state: TicketingAgentState) -> Literal["tool_executor", "END"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_executor"
        else:
            return END

    def tool_executor(state: TicketingAgentState):
        result = []
        for tc in state["messages"][-1].tool_calls:
            _id, _name, _args = tc["id"], tc["name"], tc["args"]
            tool_response = tool_name_to_executables[_name].invoke(_args)
            result.append(ToolMessage(content=tool_response, tool_call_id=_id))
        state["messages"].extend(result)
        return state

    builder = StateGraph(TicketingAgentState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_executor", tool_executor)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", check_tool_call, ["tool_executor", END])
    builder.add_edge("tool_executor", "llm_call")

    return builder.compile(checkpointer=checkpointer)
