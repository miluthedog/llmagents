import asyncio
import os
from typing import Optional
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

        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    async def connect_to_server(self, server_url: str):
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()
        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        self.tools_list = convert_mcp_tools_to_gemini(tools)

    async def process(self, user_prompt: str) -> str:
        system_prompt = """
        You are a smart assistant that solves problems using tools (functions) when necessary.
        
        Always follow this structure:
        1. First, analyze the user's request carefully and explain your reasoning step-by-step as text.
        2. If solving the problem requires a tool/function, make a function call **after your reasoning** as a separate action. Only call the function after explaining why it is needed.
        3. Once the function result is returned, use it along with your earlier reasoning to provide a complete and helpful final answer.

        Your response should contain:
        - One or more parts with your reasoning (text)
        - A separate part with the function call (if needed)
        """
            # Create JSON role 'user'
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=user_prompt)]
        )

        init_response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[user_prompt_content],
            config=types.GenerateContentConfig(
                system_instruction = system_prompt,
                tools = self.tools_list
            )
        )

        print(f"\n[DEBUG] Number of candidates: {len(init_response.candidates)}")
        for idx, cand in enumerate(init_response.candidates):
            print(f"\n[DEBUG] Candidate {idx} has {len(cand.content.parts)} parts")

        ai_response = init_response.candidates[0].content.parts
            # Create JSON role 'assistant'
        ai_response_content = types.Content(
            role='assistant',
            parts=ai_response
        )
        
        function_call_part = next((part for part in ai_response if part.function_call), None)
        if function_call_part:
            tool_name = function_call_part.function_call.name
            tool_args = function_call_part.function_call.args
            print(f"\nAgent requested tool call: {tool_name} with args {tool_args}\n")

            try:
                result = await self.session.call_tool(tool_name, tool_args)
                function_response = {"result": result.content}
            except Exception as e:
                function_response = {"error": str(e)}

                # Create JSON tool name and role 'tool'
            function_response_part = types.Part.from_function_response(
                name=tool_name,
                response=function_response
            )
            function_response_content = types.Content(
                role='tool',
                parts=[function_response_part]
            )

            new_response = self.client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[
                    user_prompt_content,
                    ai_response_content,
                    function_response_content
                ],
                config=types.GenerateContentConfig(
                    system_instruction = system_prompt,
                    tools = self.tools_list
                )
            )

            print(f"[DEBUG] User:\n{user_prompt_content}\n")
            print(f"[DEBUG] Agent:\n{ai_response_content}\n")
            print(f"[DEBUG] Tool:\n{function_response_content}")

            return new_response.candidates[0].content.parts[0].text
        else:
            return init_response.text

    async def chat_loop(self):
        while True:
            print("============")
            user_prompt = input("User: ").strip()
            if user_prompt.lower() in ['exit', 'quit']:
                break

            response = await self.process(user_prompt)
            print("\nAgent: " + response)

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)


def convert_mcp_tools_to_gemini(mcp_tools):
    gemini_tools = []

    def clean_schema(schema):
        if isinstance(schema, dict):
            schema.pop("title", None) # Recursively remove "title" key
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
    url = await asyncio.to_thread(input, "Enter SSE server URL: ")

    client = MCPClient()
    try:
        await client.connect_to_server(url)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())