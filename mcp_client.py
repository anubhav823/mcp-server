import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict
import shlex

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_mcp_adapters.tools import load_mcp_tools

# MCP server launch config
server_params = StdioServerParameters(
    command="python",
    args=["weather_server.py"]
)

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph(session):
    # Load tools from MCP server
    tools = await load_mcp_tools(session)

    # LLM configuration 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key="<GOOGLE_API_KEY>" )
    llm_with_tools = llm.bind_tools(tools)

    # Prompt template with user/assistant chat only
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to get the current weather for a location."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Build LangGraph with tool routing
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

async def list_prompts(session):
    try:
        prompt_response = await session.list_prompts()
        if not prompt_response or not prompt_response.prompts:
            print("No prompts found on the MCP server.")
            return
        print("Available prompts on the MCP server:")
        for prompt in prompt_response.prompts:
            print(f"- {prompt.name}")
            if prompt.arguments:
                arg_list = [f"<{arg.name}>" for arg in prompt.arguments]
                print(f"Arguments: {', '.join(arg_list)}")
            else:
                print("Arguments: None")
        print("\nUsage: /prompt <prompt_name> \"arg1\" \"arg2\" ...")
        print("---------------------------------------")
    except Exception as e:
        print(f"Error fetching prompts: {e}")

async def handle_prompt(session, command):
    try:
        parts = shlex.split(command)
        if len(parts) < 2:
            print("Usage: /prompt <prompt_name> \"arg1\" \"arg2\" ...")
            return
        prompt_name = parts[1]
        args = parts[2:]
        prompt_def_response = await session.list_prompts()
        if not prompt_def_response or not prompt_def_response.prompts:
            print("No prompts found on the MCP server.")
            return
        prompt_def = next((p for p in prompt_def_response.prompts if p.name == prompt_name), None)
        if not prompt_def:
            print(f"Prompt '{prompt_name}' not found on the MCP server.")
            return None
        if len(args) != len(prompt_def.arguments):
            expected_args = len(prompt_def.arguments)
            print(f"\nError: Prompt '{prompt_name}' expects {expected_args} arguments, but {len(args)} were provided.")
            return None
        arg_dict = {arg.name: val for arg, val in zip(prompt_def.arguments, args)}

        prompt_response = await session.get_prompt(prompt_name, arg_dict)

        prompt_text = prompt_response.messages[0].content.text
        print("\nPrompt loaded. Getting ready to execute...")
        return prompt_text

    except Exception as e:
        print(f"Error parsing command: {e}")
        return


# Entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            
            print("Weather MCP agent is ready.")
            print("Type a question, or use one of the following commands:")
            print("  /prompts                           - to list available prompts")
            print("  /prompt <prompt_name> \"args\"...  - to run a specific prompt")
            print("  /resources                       - to list available resources")
            print("  /resource <resource_uri>         - to load a resource for the agent")  
            while True:
                message_to_agent = ""
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                
                if user_input.lower() == "/prompts":
                    await list_prompts(session)
                    continue
                elif user_input.lower().startswith("/prompt"):
                    prompt_text = await handle_prompt(session, user_input)
                    if prompt_text:
                        message_to_agent = prompt_text
                    else:
                        continue
                elif user_input.lower() == "/resources":
                    await list_resources(session)
                    continue
                elif user_input.lower().startswith("/resource"):
                    resource_content = await handle_resource(session, user_input)
                    if resource_content:
                        action_prompt = input('Resource loaded. What to do?').strip().lower()
                        if action_prompt:
                            message_to_agent = f"""
                            CONTEXT from a loaded resource:
                            ---
                            {resource_content}
                            ---
                            TASK: {action_prompt}
                            """
                        else:
                            print("No action specified. Adding resource content to conversation memory...")
                            message_to_agent = f"""
                            Please remember the following context for our conversation. Just acknowledge that you have received it.
                            ---
                            CONTEXT:
                            {resource_content}
                            ---
                            """
                    else:
                        # If resource loading failed, loop back for next input
                        continue
                else:
                    # For a normal chat message, the message is just the user's input
                    message_to_agent = user_input

                if message_to_agent:
                    try:
                        response = await agent.ainvoke({"messages": [("user", message_to_agent)]},
                            config={"configurable": {"thread_id": "weather-session"}}
                        )
                        print("AI:", response["messages"][-1].content)
                    except Exception as e:
                        print("Error:", e)

async def list_resources(session):
    try:
        resource_response = await session.list_resources()
        if not resource_response or not resource_response.resources:
            print("No resources found on the MCP server.")
            return
        for resource in resource_response.resources:
            print(f'Resource uri: {resource.uri}')
            if resource.description:
                print(f'  Description: {resource.description.strip()}')
    except Exception as e:
        print(f"Error fetching resources: {e}")

async def handle_resource(session, command) -> str|None:
    try:
        parts = shlex.split(command)
        if len(parts) != 2:
            print("Usage: /resource <resource_uri>")
            return None
        
        resource_uri = parts[1]
        print(f"\n--- Fetching resource '{resource_uri}'... ---")

        resource_response = await session.read_resource(resource_uri)
        if not resource_response or not resource_response.contents:
            print(f"Resource not found for '{resource_uri}'.")
            return None 
        text_parts = [content.text for content in resource_response.contents if hasattr(content, "text")]
        if not text_parts:
            print(f"Resource '{resource_uri}' does not contain text content.")
            return None
        resource_content = '\n'.join(text_parts)
        print('Loaded successfully')
        return resource_content
    except Exception as e:
        print(f"Error fetching resource: {e}")

if __name__ == "__main__":
    asyncio.run(main())