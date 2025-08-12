import asyncio
import os
import json
import subprocess
from typing import Dict, List, Optional
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
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


    async def connect_to_sse_server(self, server_url: str, server_id: str):
        print(f"Connecting to SSE server [{server_id}]: {server_url}")
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
            print(f"SSE Server [{server_id}] connected with tools: {[tool.name for tool in tools]}")

            server_tools = convert_mcp_tools_to_gemini(tools)
            self.tools_list.extend(server_tools)
            for tool in tools:
                self.tool_to_server_mapping[tool.name] = server_id
                
            return True
            
        except Exception as e:
            print(f"Error connecting to SSE server [{server_id}]: {e}")
            return False


    async def connect_to_subprocess_server(self, command: str, args: List[str], env: Dict[str, str], server_id: str):
        print(f"Starting subprocess server [{server_id}]: {command} {' '.join(args)}")
        try:
            full_env = os.environ.copy()
            full_env.update(env)
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=full_env
            )
            
            streams_context = stdio_client(server_params)
            streams = await streams_context.__aenter__()
            session_context = ClientSession(*streams)
            session = await session_context.__aenter__()
            await session.initialize()

            self.streams_contexts[server_id] = streams_context
            self.session_contexts[server_id] = session_context
            self.sessions[server_id] = session

            response = await session.list_tools()
            tools = response.tools
            print(f"Subprocess server [{server_id}] connected with tools: {[tool.name for tool in tools]}")

            server_tools = convert_mcp_tools_to_gemini(tools)
            self.tools_list.extend(server_tools)
            for tool in tools:
                self.tool_to_server_mapping[tool.name] = server_id
                
            return True
            
        except Exception as e:
            print(f"Error connecting to subprocess server [{server_id}]: {e}")
            return False


    async def connect_to_server(self, server_config: Dict, server_id: str):
        if 'url' in server_config: # Local SSE server
            return await self.connect_to_sse_server(server_config['url'], server_id)
        elif 'command' in server_config: # Subprocess server
            command = server_config['command']
            args = server_config.get('args', [])
            env = server_config.get('env', {})
            return await self.connect_to_subprocess_server(command, args, env, server_id)
        else:
            print(f"Invalid server configuration")
            return False


    async def connect_to_multiple_servers(self, server_configs: List[Dict]):
        connection_tasks = []
        for config in server_configs:
            server_id = config.get('id') or config.get('name', f"server_{len(connection_tasks)}")
            task = self.connect_to_server(config, server_id)
            connection_tasks.append(task)
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = sum(1 for result in results if result is True)
        print(f"\nSuccessfully connected to {successful_connections}/{len(server_configs)} servers")
        print(f"Total available tools: {len(self.tools_list)}")


    async def execute_function_calls(self, function_call_parts: List) -> List:
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

    async def process(self, user_prompt: str) -> str:
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
            
            function_call_parts = [part for part in ai_response_parts if hasattr(part, 'function_call') and part.function_call]
            
            if function_call_parts:
                print(f"Agent requested {len(function_call_parts)} tool call(s)")
                function_response_parts = await self.execute_function_calls(function_call_parts)
                function_response_content = add_json_role('tool', function_response_parts)
                conversation_history.append(function_response_content)
                continue
            else:
                final_text = response.text if response.text else "Task completed."
                return final_text

        return response.text


    async def chat_loop(self):
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
            cleaned = schema.copy()

            for prop in ["title", "$schema", "additionalProperties", "additional_properties"]: # top-level keys
                cleaned.pop(prop, None)

            dict_keys = ["properties", "definitions"] # dict of schemas
            for key in dict_keys:
                if key in cleaned and isinstance(cleaned[key], dict):
                    cleaned[key] = {k: clean_schema(v) for k, v in cleaned[key].items()}

            list_keys = ["allOf", "anyOf", "oneOf"] # list of schemas
            for key in list_keys:
                if key in cleaned and isinstance(cleaned[key], list):
                    cleaned[key] = [clean_schema(item) for item in cleaned[key]]

            if "items" in cleaned: # single schema
                cleaned["items"] = clean_schema(cleaned["items"])
            return cleaned

        elif isinstance(schema, list):
            return [clean_schema(item) for item in schema]
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
        with open("config.json", "r") as f:
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