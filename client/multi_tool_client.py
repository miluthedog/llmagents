import asyncio
import os
import json
from typing import Dict, List
from mcp import ClientSession
from mcp.client.sse import sse_client
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

class MCPClient:
    def __init__(self):
        self.client = genai.Client(api_key=gemini_api_key)
        
        self.sessions: Dict[str, ClientSession] = {}
        self.streams_contexts: Dict[str, any] = {}
        self.session_contexts: Dict[str, any] = {}
        
        self.tools_list = []
        self.tool_to_server_mapping = {}
        self.max_turns = 5

    async def connect_to_server(self, server_url: str, server_id: str): # connect to a server
        print(f"Connecting to server [{server_id}]: {server_url}")
        try:
            streams_context = sse_client(url=server_url)
            streams = await streams_context.__aenter__()
            session_context = ClientSession(*streams)
            session = await session_context.__aenter__()
            await session.initialize()

            self.streams_contexts[server_id] = streams_context
            self.session_contexts[server_id] = session_context
            self.sessions[server_id] = session

            response = await session.list_tools()
            tools = response.tools
            print(f"Server [{server_id}] connected with tools: {[tool.name for tool in tools]}")

            server_tools = convert_mcp_tools_to_gemini(tools)
            self.tools_list.extend(server_tools)
            for tool in tools:
                self.tool_to_server_mapping[tool.name] = server_id
                
            return True
            
        except Exception as e:
            print(f"Error connect to server [{server_id}]: {e}")
            return False


    async def connect_to_multiple_servers(self, server_configs: List[Dict[str, str]]): # asyncio.gather call connect_in_server()
        connection_tasks = []
        for config in server_configs:
            task = self.connect_to_server(config['url'], config['id'])
            connection_tasks.append(task)
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = sum(1 for result in results if result is True)
        print(f"\nSuccessfully connected to {successful_connections}/{len(server_configs)} servers")
        print(f"Total available tools: {len(self.tools_list)}")


    async def execute_function_calls(self, function_call_parts: List) -> List: # execute all function calls
        function_response_parts = []
        for function_call_part in function_call_parts:
            tool_name = function_call_part.function_call.name
            tool_args = function_call_part.function_call.args

            server_id = self.tool_to_server_mapping.get(tool_name)
            if not server_id:
                function_response = {"error": f"Tool '{tool_name}' not found in any connected server"}
                print(f"ERROR: Tool '{tool_name}' not found in any server")
            else:
                print(f"Calling tool: {tool_name} (from server {server_id}) with args {tool_args}")
                
                try:
                    session = self.sessions[server_id]
                    result = await session.call_tool(tool_name, tool_args)
                    function_response = {"result": result.content}
                    print(f"Tool {tool_name} completed successfully")
                except Exception as e:
                    function_response = {"Error": str(e)}

            function_response_part = types.Part.from_function_response(
                name=tool_name,
                response=function_response
            )
            function_response_parts.append(function_response_part)
        
        return function_response_parts


    async def process(self, user_prompt: str) -> str: # agent loop answer (execute tool or not -> call function)
        system_prompt = """
        You are a smart assistant with access to tools on multiple servers.

        You have a limit of **Turn**(either text, function call or both) per task. Plan carefully.

        Your job:
        1. Understand the user's request fully before acting by analysing it step-by-step.
        2. Use tools in **parallel** when tasks are independent.
        3. Use **sequential** calls only when one result depends on another.
        4. Avoid unnecessary steps — combine or batch operations when possible.
        5. Think before executing. Finish the task **accurately** and **within the limit**.

        Bad examples:
        - Breaking simple tasks into too many steps
        - Using 3 turns for 1 + 2 + 3
        - Good: 1 + 2 → + 3 → Final answer (2 turns)
        - Best: If supported, do all at once (1 turn)

        Always minimize turns. Finish the task correctly.
        """
        user_prompt_content = add_json_role('user', user_prompt)
        conversation_history = [user_prompt_content]

        turn_count = 0
        while turn_count < self.max_turns:
            turn_count += 1
            print(f"\n=== Turn {turn_count} ===")
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=self.tools_list
                )
            )

            ai_response_parts = response.candidates[0].content.parts
            ai_response_content = add_json_role('assistant', ai_response_parts)
            conversation_history.append(ai_response_content)
            
            function_call_parts = [part for part in ai_response_parts if part.function_call]
            
            if function_call_parts:
                print(f"Agent requested {len(function_call_parts)} tool call(s)")
                function_response_parts = await self.execute_function_calls(function_call_parts)
                function_response_content = add_json_role('tool', function_response_parts)
                conversation_history.append(function_response_content)
                continue
            else:
                final_text = response.text
                return final_text

        return response.text


    async def chat_loop(self): # UI
        print(f"\nAvailable tools from all servers:")
        for tool_name, server_id in self.tool_to_server_mapping.items():
            print(f"  - {tool_name} (from {server_id})")
  
        while True:
            print("="*50)
            user_prompt = input("User: ").strip()
            if user_prompt.lower() in ['exit', 'quit']:
                break

            response = await self.process(user_prompt)
            print("\nAgent: " + response)


    async def cleanup(self):
        for server_id in list(self.session_contexts.keys()):
            try:
                if self.session_contexts[server_id]:
                    await self.session_contexts[server_id].__aexit__(None, None, None)
                if self.streams_contexts[server_id]:
                    await self.streams_contexts[server_id].__aexit__(None, None, None)
            except Exception as e:
                print(f"Error cleanup server {server_id}: {e}")


def add_json_role(role: str, parts) -> types.Content:
    if isinstance(parts, str):
        parts = [types.Part.from_text(text=parts)]
    return types.Content(role=role, parts=parts)


def convert_mcp_tools_to_gemini(mcp_tools):
    gemini_tools = []

    def clean_schema(schema):
        if isinstance(schema, dict):
            schema.pop("title", None)  # Recursively remove "title" key
            if "properties" in schema and isinstance(schema["properties"], dict):
                for key in schema["properties"]:
                    schema["properties"][key] = clean_schema(schema["properties"][key])
        return schema

    for tool in mcp_tools:
        parameters = clean_schema(tool.inputSchema)

        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )

        gemini_tool = Tool(function_declarations=[function_declaration])
        gemini_tools.append(gemini_tool)
    return gemini_tools


async def main():
    client = MCPClient()

    try:
        with open("server_config.json", "r") as f:
            config_data = json.load(f)
            server_configs = config_data.get("servers", [])
            print(f"Loaded {len(server_configs)} server configurations\n")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error JSON: {e}")

    try:
        await client.connect_to_multiple_servers(server_configs)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())