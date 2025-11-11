from os import getenv

import json
import uuid

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union, Dict, List, Any, Optional

# MLflow imports for the ResponsesAgent interface
from mlflow.pyfunc import ResponsesAgent, PythonModelContext 
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import Message

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI


# LangGraph imports
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable 
from langgraph.types import Command, Interrupt

WRAP_INTERRUPT = "__interrupt__"
WRAP_RESUME = "__resume__"

def _mlflow_input_message_to_langchain_message(msg: Union[Dict[str, Any], Message]) -> BaseMessage:
        """
        Converts an MLflow input message (which can be a dict or a 
        mlflow.types.responses_helpers.Message object) to a LangChain BaseMessage.
        """
        
        role: str = "user"
        content: str = ""
        tool_calls: Optional[List[Dict[str, Any]]] = None
        tool_call_id: str = ""  # ToolMessage requires a non-None string

        if isinstance(msg, dict):
            # Handle dictionary input
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id", "")
        
        elif isinstance(msg, Message):
            # Handle Message object input (using getattr for safety)
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)
            tool_call_id = getattr(msg, "tool_call_id", "")
            
        else:
            # Fallback for any other unexpected type
            content = str(msg)
            # All other fields (role, tool_calls, tool_call_id)
            # will use their safe defaults defined above.

        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            # Pass tool_calls only if it's not None and not empty
            if tool_calls:
                return AIMessage(content=content, tool_calls=tool_calls)
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        if role == "tool":
            # tool_call_id is guaranteed to be a string
            return ToolMessage(content=content, tool_call_id=tool_call_id)
            
        # Default to HumanMessage
        return HumanMessage(content=content)

class LanggraphResponsesAgent(ResponsesAgent, ABC):
    """
    An abstract base class (ABC) that wraps a LangGraph agent to conform to the
    mlflow.pyfunc.ResponsesAgent interface.

    This class handles the translation between MLflow's ResponsesAgentRequest/Response
    and LangGraph's input/output formats, including the specific interrupt and
    resume logic via `custom_inputs` and `custom_outputs`.

    Subclasses must implement the `get_graph` method.
    """

    def __init__(self):
        """
        Initializes the agent. Key components like the graph, llm, and
        checkpointer will be initialized in the `load_context` method.
        """
        self.graph: Runnable  
        self.checkpointer: BaseCheckpointSaver
        self.llm: BaseChatModel
        self.agent_tools: List[Any] = []

    @abstractmethod
    def get_graph(
        self,
        llm: BaseChatModel,
        checkpointer: BaseCheckpointSaver,
        agent_tools: List[Any],
    ) -> Runnable:
        """
        Abstract method for child classes to implement.

        This method must construct and return a compiled LangGraph application
        using the provided components.

        Args:
            llm: The BaseChatModel instance to be used by the agent.
            checkpointer: The checkpoint saver instance for managing state.
            agent_tools: A list of tools to be used by the agent.

        Returns:
            A compiled LangGraph runnable 
        """
        pass

    def get_agent_tools(self, llm_factory, checkpointer_factory) -> List[Runnable]:
        """
        Loads and returns the list of agent tools.

        This default implementation returns an empty list.
        Subclasses can override this method to load tools
        from the context if needed.
        """
        return []

    def load_context(self, context: PythonModelContext):
        """
        Initializes the agent's graph and components.

        This method is called by MLflow when the model is loaded. It
        orchestrates the creation of the checkpointer, loading of the
        LLM and tools, and the final construction of the graph by
        calling the abstract `get_graph` method.
        """

        def llm_factory():
            return ChatOpenAI(
                api_key=getenv("OPENAI_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model="gpt-4o-mini",
                )

        def checkpointer_factory():
            return InMemorySaver()

        self.llm = llm_factory()
        self.checkpointer = checkpointer_factory()
        self.agent_tools = self.get_agent_tools(llm_factory, checkpointer_factory)      
        
        self.graph = self.get_graph(
            llm=self.llm,
            checkpointer=self.checkpointer,
            agent_tools=self.agent_tools
        )
        print("LanggraphResponsesAgent loaded successfully.")

    # --- Overridable Hooks ---

    def preprocess_request(
        self,
        request: ResponsesAgentRequest,
        langgraph_input: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Overridable hook to modify the LangGraph input and config
        before .invoke() or .stream() is called.

        A key use case is to inspect `request.custom_inputs[WRAP_RESUME]`
        and format the `langgraph_input` payload correctly, e.g.,
        as a ToolMessage instead of a HumanMessage.

        Args:
            request: The raw ResponsesAgentRequest.
            langgraph_input: The generated input payload for LangGraph.
            config: The generated config for LangGraph.

        Returns:
            A tuple of (modified_langgraph_input, modified_config).
        """
        return langgraph_input, config

    def postprocess_response(
        self,
        langgraph_output: Dict[str, Any],
        response: Union[ResponsesAgentResponse, ResponsesAgentStreamEvent],
    ) -> Union[ResponsesAgentResponse, ResponsesAgentStreamEvent]:
        """
        Overridable hook to modify the final ResponsesAgentResponse
        (for predict) or the final ResponsesAgentStreamEvent (for predict_stream)
        before it is returned to the client.

        Args:
            langgraph_output: The raw state dict from LangGraph.
            response: The generated MLflow response object.

        Returns:
            The (potentially modified) MLflow response object.
        """
        return response

    # --- Internal Helper Methods ---

    def _build_input_payload(self, request: ResponsesAgentRequest) -> Dict[str, Any]:
        """
        Builds the input dictionary for LangGraph based on the request,
        handling both new turns and resume signals.
        """
        if request.custom_inputs and WRAP_RESUME in request.custom_inputs:
            # Resuming from an interrupt.
            # The WRAP_RESUME value is the JSON-serialized data
            # sent by the client, which we assume is the human's
            # response to the agent's question.
            try:
                resume_data = json.loads(request.custom_inputs[WRAP_RESUME])
            except (json.JSONDecodeError, TypeError):
                # Fallback if it's not JSON but just a plain string
                resume_data = request.custom_inputs[WRAP_RESUME]

            langgraph_input = Command(resume=resume_data) 
        else:
            # New turn. Convert request.input to LangChain messages.
            langgraph_input = {
                "messages": [_mlflow_input_message_to_langchain_message(msg) for msg in request.input]
            }
        
            if "loop_counter" not in langgraph_input:
                langgraph_input["loop_counter"] = 0
            
        return langgraph_input

    def _get_config(self, request: ResponsesAgentRequest) -> Dict[str, Any]:
        """Builds the LangGraph config dict with the thread_id."""
        return {"configurable": {"thread_id": request.context.conversation_id}}

    def _serialize_interrupt_data(self, interrupt_data: Any) -> str:
        """Serializes interrupt data to a JSON string per requirements."""
        try:
            return json.dumps([{"value": id.value, "id": id.id} for id in interrupt_data])
        except TypeError as e:
            # Fallback for non-serializable data (e.g., complex objects)
            return json.dumps(str(interrupt_data))

    def _get_interrupt_text(self, interrupt_data: Any) -> str:
        """Gets the string representation of the interrupt for display."""
        if isinstance(interrupt_data, (str, int, float, bool)):
            return str(interrupt_data)
        if isinstance(interrupt_data, dict):
            try:
                # Try to pretty-print if it's a dict
                return json.dumps(interrupt_data, indent=2)
            except TypeError:
                return str(interrupt_data)
        return str(interrupt_data)

    # --- ResponsesAgent Implementation ---

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Handles synchronous, non-streaming requests.
        """
        # 1. Prepare input and config
        config = self._get_config(request)
        langgraph_input = self._build_input_payload(request)

        # 2. Pre-Hook
        langgraph_input, config = self.preprocess_request(
            request, langgraph_input, config
        )

        # 3. Run
        final_state = self.graph.invoke(langgraph_input, config=config)

        # 4. Handle Output
        custom_outputs = {}
        output_item_id = str(uuid.uuid4())

        if WRAP_INTERRUPT in final_state:
            # Agent interrupted, waiting for input
            interrupt_data = final_state[WRAP_INTERRUPT]
            custom_outputs[WRAP_INTERRUPT] = self._serialize_interrupt_data(interrupt_data)
            output_text = self._get_interrupt_text(interrupt_data) # this will get overriden if there are messages
            output_item = self.create_text_output_item(
                text=output_text, id=output_item_id
            )
        else:
            # Agent finished normally
            if "messages" in final_state and final_state["messages"]:
                final_message = final_state["messages"][-1]
                output_text = str(final_message.content)
            else:
                output_text = "[Agent finished without a final message]"
            
            output_item = self.create_text_output_item(
                text=output_text, id=output_item_id
            )

        response = ResponsesAgentResponse(
            output=[output_item], custom_outputs=custom_outputs
        )

        # 5. Post-Hook
        return self.postprocess_response(final_state, response)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Iterator[ResponsesAgentStreamEvent]:
        """
        Handles streaming requests. NOT PROPERLY TESTED YET.
        """
        # 1. Prepare input and config
        config = self._get_config(request)
        langgraph_input = self._build_input_payload(request)

        # 2. Pre-Hook
        langgraph_input, config = self.preprocess_request(
            request, langgraph_input, config
        )

        # 3. Run
        stream = self.graph.stream(langgraph_input, config=config)

        last_full_state = {}
        streamed_message_ids = set()
        current_message_chunks = {}  # Stores item_id -> content

        for _chunk in stream:
            # Update the last known full state
            last_full_state.update(_chunk)
            
            # --- A. Handle Message Streaming ---
            if WRAP_INTERRUPT not in _chunk.keys():
                chunk = list(_chunk.values())[0] # assumption is single item dict
                # LangGraph streams the *entire* list of messages
                # We only care about the latest one and if it's an AIMessage
                latest_message = chunk["messages"][-1]
                
                if isinstance(latest_message, AIMessage) and latest_message.content:
                    item_id = str(latest_message.id or uuid.uuid4())

                    if item_id not in streamed_message_ids:
                        # First chunk for this message
                        streamed_message_ids.add(item_id)
                        delta = latest_message.content
                        current_message_chunks[item_id] = delta
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=delta, item_id=item_id)
                        )
                    else:
                        # Subsequent chunk for this message
                        full_content = latest_message.content
                        prev_content = current_message_chunks.get(item_id, "")

                        if full_content != prev_content:
                            # Calculate and yield the delta
                            delta = full_content[len(prev_content) :]
                            current_message_chunks[item_id] = full_content
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(delta=delta, item_id=item_id)
                            )

            # --- B. Handle Interrupt ---
            if WRAP_INTERRUPT in _chunk:
                # An interrupt has occurred.
                print ("interrupt detected in stream.")
                # Yield "done" events for any in-progress messages
                for item_id, content in current_message_chunks.items():
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(text=content, id=item_id),
                    )
                
                # Create and yield the FINAL interrupt event
                interrupt_data = _chunk[WRAP_INTERRUPT]
                interrupt_id = interrupt_data[0].id if isinstance(interrupt_data, list) and len(interrupt_data) > 0 else str(uuid.uuid4())
                
                custom_outputs = {
                    WRAP_INTERRUPT: self._serialize_interrupt_data(interrupt_data)
                }
                output_text = self._get_interrupt_text(interrupt_data)
                
                item = self.create_text_output_item(text=output_text, id=interrupt_id)
                
                final_event = ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=item,
                    custom_outputs=custom_outputs,
                )
                
                # 5. Post-Hook (for the interrupt event)
                yield self.postprocess_response(_chunk, final_event)
                
                # Stop the generator as required
                return

        # --- C. Handle Normal Finish (No interrupt) ---
        
        # Send "done" events for all messages that were streamed
        for item_id, content in current_message_chunks.items():
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=content, id=item_id),
            )
            
        # If no messages were streamed (e.g., graph finished instantly
        # or only had tool calls), we send the *last* message from the
        # final state as a single "done" event.
        if not streamed_message_ids:
            final_message = None
            if "messages" in last_full_state and last_full_state["messages"]:
                # Get the last message, which should be the final response
                final_message = last_full_state["messages"][-1]

            # Only send if it's an AIMessage with content
            if isinstance(final_message, AIMessage) and final_message.content:
                item_id = str(final_message.id or uuid.uuid4())
                content = final_message.content
                item = self.create_text_output_item(text=content, id=item_id)
                
                final_event = ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=item,
                    custom_outputs={}
                )
                # 5. Post-Hook (for the final event)
                yield self.postprocess_response(last_full_state, final_event)