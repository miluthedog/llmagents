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

    async def process(self, prompt: str) -> str:
        system_prompt = """
        You are an AI assistant that can use external tools (functions) to help answer questions or perform tasks.

        - When needed, respond with a function call using the tools you have been provided.
        - If the answer is already known and no tool is required, respond normally.
        - Your goal is to use tools efficiently and clearly, then integrate results into helpful answers.
        """

        user_prompt = types.Content(
            role='user',
            parts=[types.Part.from_text(text=prompt)]
        )

        responses = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction = system_prompt,
                tools = self.tools_list
            )
        )

        response = responses.candidates[0].content.parts[0]
        if response.function_call:
            tool_name = response.function_call.name
            tool_args = response.function_call.args
            print(f"\n[Agent requested tool call: {tool_name} with args {tool_args}]")

            try:
                result = await self.session.call_tool(tool_name, tool_args)
                function_response = {"result": result.content}
            except Exception as e:
                function_response = {"error": str(e)}

                # Create JSON title for tool name
            function_response_part = types.Part.from_function_response(
                name=tool_name,
                response=function_response
            )
                # Create JSON role
            function_response_content = types.Content(
                role='tool',
                parts=[function_response_part]
            )

            contents=[
                    user_prompt,
                    response,
                    function_response_content
                ]
            print(contents)

            new_response = self.client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[
                    user_prompt,
                    response,
                    function_response_content
                ],
                config=types.GenerateContentConfig(
                    system_instruction = system_prompt,
                    tools = self.tools_list
                )
            )
            return new_response.candidates[0].content.parts[0].text
        else:
            return response.text

    async def chat_loop(self):
        while True:
            prompt = input("\User: ").strip()
            if prompt.lower() == 'exit' or prompt.lower() == 'quit':
                break

            response = await self.process(prompt)
            print("\n" + response)

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