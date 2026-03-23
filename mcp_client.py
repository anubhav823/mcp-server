import asyncio
import shlex
from mcp import StdioServerParameters
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Multi-server configuration dictionary ---
# This dictionary defines all the servers the client will connect to
server_configs = {
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],  # The weather server
        "transport": "stdio",
    },
    "tasks": {
        "command": "python",
        "args": ["task_server.py"], # The new task management server
        "transport": "stdio",
    }
}

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# --- 'create_graph' now accepts the list of tools directly ---
def create_graph(tools: list):
    # LLM configuration 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key="<>" )
    llm_with_tools = llm.bind_tools(tools)

    # --- Updated system prompt to reflect new capabilities ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have access to tools for checking the weather and managing a to-do list. Use the tools when necessary based on the user's request."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node 
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

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

# Iterates through all configured servers, lists their available resources,
# and prints them in a user-friendly format
async def list_all_resources(client: MultiServerMCPClient, server_configs: dict):
    print("\nAvailable Resources from all servers:")
    print("-------------------------------------")
    
    any_resources_found = False
    
    # Iterate through the names of the servers defined in your server_configs
    for server_name in server_configs.keys():
        try:
            # Opening a session for a specific server to list its resources
            async with client.session(server_name) as session:
                # The method to list resources is session.list_resources()
                resource_response = await session.list_resources()

                if resource_response and resource_response.resources:
                    any_resources_found = True
                    print(f"\n--- Server: '{server_name}' ---")
                    for r in resource_response.resources:
                        # The most important identifier for a resource is its URI
                        print(f"  Resource URI: {r.uri}")
                        if r.description:
                            print(f"    Description: {r.description}")
        except Exception as e:
            print(f"\nCould not fetch resources from server '{server_name}': {e}")
    
    print("\nUse: /resource <server_name> <resource_uri>")
    print("-----------------------------------")          
    
    if not any_resources_found:
        print("\nNo resources were found on any connected servers.")

# Parses a user command to fetch a specific resource from a server and
# returns its content as a string
async def handle_resource_invocation(client: MultiServerMCPClient, command: str) -> str | None:
    try:
        
        parts = command.strip().split()
        if len(parts) != 3:
            print("\nUsage: /resource <server_name> <resource_uri>")
            return None

        server_name = parts[1]
        resource_uri = parts[2]

        print(f"\n--- Fetching resource '{resource_uri}' from server '{server_name}'... ---")

        # It requires the server_name and the URI of the resource
       
        blobs = await client.get_resources(server_name=server_name, uris=[resource_uri])

        if not blobs:
            print("Error: Resource not found or content is empty.")
            return None

        # Converting LangChain Blobs to string content
       
        resource_content = blobs[0].as_string()
        
        print("--- Resource content loaded successfully. ---")
        return resource_content

    except Exception as e:
        print(f"\nAn error occurred while fetching the resource: {e}")
        return None
    
# --- Main function ---
async def main():

    # The client will manage the server subprocesses internally
    client = MultiServerMCPClient(server_configs)
    
    # Get a single, unified list of tools from all connected servers
    all_tools = await client.get_tools()

    # Create the LangGraph agent with the aggregated list of tools
    agent = create_graph(all_tools)
    
    print("MCP Agent is ready (connected to Weather and Task servers).")
    print("Type a question, or use one of the following commands:")
    print("  /resources                             - to list available resources")
    print("  /resource <server_name> <resource_uri>      - to load a resource for the agent")
    
    message_to_agent = ""
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if user_input.lower() == "/resources":
            await list_all_resources(client, server_configs)
            continue
        elif user_input.startswith("/resource"):
            resource_content = await handle_resource_invocation(client, user_input)

            if resource_content:
                action_prompt = input("Resource loaded. What should I do with this content? (Press Enter to just save to context)\n> ").strip()
                
                # If user provides an action, combine it with the resource content
                if action_prompt:
                    message_to_agent = f"""
                    CONTEXT from a loaded resource:
                    ---
                    {resource_content}
                    ---
                    TASK: {action_prompt}
                    """
              
                # If user provides no action, create a default message to save the context
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

        # Final agent invocation
        if message_to_agent:
            try:
                response = await agent.ainvoke(
                    {"messages": [("user", message_to_agent)]},
                    config={"configurable": {"thread_id": "multi-server-session"}}
                )
                print("AI:", response["messages"][-1].content)
            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())