from uuid import uuid4
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt

def create_ticketing_agent_tool(agent_tkt):
    @tool(
        "ticketing_agent_tool",
        description="A tool for interacting with the ticketing agent. \
            Use this tool to delegate ticketing-related queries to the ticketing agent. \
            Make sure you provide context information to the agent, so it has the full context.",
    )
    def agent_tkt_tool(query: str) -> str:
        result = {
                "ticket_booked": False,
                "messages": [],
                "loop_counter": 0,
            } 
        config = {"configurable": {"thread_id": str(uuid4()) }}
        while not result["ticket_booked"]:
            if result["first_invoked"]:
                caller_message = HumanMessage(content=query)
                result["first_invoked"] = False
            else:
                question = str(result["messages"][-1].content)
                caller_content = interrupt("Ticketing Agent has a question: \n" + question + "\nYour response: ")
                caller_message = HumanMessage(content=caller_content)
            result["messages"].append(caller_message)
                
            result = agent_tkt.invoke(result, config=config)
            
            while "__interrupt__" in result:
                question = str(result["__interrupt__"][0].value)
                caller_response = interrupt("Ticketing Agent has a question: \n" + question + "\nYour response: ")
                caller_message = HumanMessage(content=caller_response)
                result["messages"].append(caller_message)
                result = agent_tkt.invoke(Command(resume=caller_response), config=config)
            return result["messages"][-1].content
    
    return agent_tkt_tool