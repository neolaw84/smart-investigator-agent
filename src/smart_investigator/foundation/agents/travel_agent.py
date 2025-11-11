
from typing import List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph, CompiledStateGraph, START, END
from langchain_core.runnables import Runnable

from smart_investigator.foundation.utils.langgraph_responses_agent import LanggraphResponsesAgent
from smart_investigator.foundation.agents.travel_agent_graph import create_travel_agent
from smart_investigator.foundation.agents.ticketing_agent_graph import create_ticketing_agent   

class TravelAgentWrapper(LanggraphResponsesAgent):
    """
    A concrete implementation of LanggraphResponsesAgent that wraps
    the 'create_travel_agent' graph from the smart-investigator project.
    """
    def get_agent_tools(self, llm_factory, checkpointer_factory):
        return [create_ticketing_agent(
            llm=llm_factory(),
            checkpointer=checkpointer_factory(),
            agent_tools=[],
        )]

    def get_graph(
        self,
        llm: BaseChatModel,
        checkpointer: BaseCheckpointSaver,
        agent_tools: List[Any],
    ) -> Runnable:
        """
        Concrete implementation of the abstract method.

        This method constructs and returns the compiled Ticketing Agent graph.
        """
        return create_ticketing_agent(
            llm=llm,
            checkpointer=checkpointer,
            agent_tools=agent_tools,
        )
