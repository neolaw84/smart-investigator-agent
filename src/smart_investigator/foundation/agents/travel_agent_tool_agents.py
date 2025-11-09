from uuid import uuid4
from typing import Annotated
from langchain.tools import InjectedToolCallId
from langchain.tools import tool
from langgraph.types import Command, interrupt

def create_ticketing_agent_tool(agent_tkt):
    @tool(
        "ticketing_agent_tool",
        description="A tool for interacting with the ticketing agent. \
            Use this tool to delegate ticketing-related queries to the ticketing agent. \
            Make sure you provide what you have to the agent (e.g., number of tickets, \
            round trip or one way, origin and destination) so it has the full context.",
    )
    def agent_tkt_tool(query: str) -> str:
        result = agent_tkt.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        while "__interrupt__" in result:
            human_response = interrupt("Ticketing Agent has a question: \n" + str(result["messages"][-1].content) + "\nYour response: ")
            config = {"configurable": {"thread_id": str(uuid4()) }}
            result = agent_tkt.invoke(Command(resume=human_response), config=config)
        return result["messages"][-1].content
    
    return agent_tkt_tool