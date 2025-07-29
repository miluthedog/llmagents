# from viewer
import os
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv
load_dotenv()


def extract_action(text):
    match = re.search(r"<calculator>(.*?)</calculator>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_calculator(expression: str) -> str:
    try:
        print('[LOG] LLM sử dụng tool để tính toán', expression)
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def generate():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    model = "gemini-2.0-flash"

    system_instruction = types.Part.from_text(text="""Bạn là một AI có thể gọi công cụ \"Calculator\" để thực hiện các phép tính.
Khi người dùng yêu cầu tính toán hoặc đưa ra biểu thức toán học, hãy lần lượt thực hiện theo chu trình ReAct:

1. Thought: Suy nghĩ xem cần tính gì
2. Action: <calculator>biểu thức</calculator>
3. Observation: Kết quả từ công cụ
4. Lặp lại nếu cần đến khi đưa ra Answer cuối cùng.
Hành động theo lượt như sau
- Model sẽ trả về Thought, Action
- User sẽ đưa ra Observation
- Model tiếp tục hành động cho đến khi đưa ra kết quả
# Ví dụ
<example>
<user_query>
Tính tổng 100 và 200
</user_query>
<assistant_response>
Thought: Sử dụng tool caculator để tính 100 và 200
Action: <calculator>100 + 200</calculator>
</assistant_response>
<user_query>
Observation: 300
</user_query>
<assistant_response>
Thought: Tổng 100 và 200 là 300
Answer: 300
</assistant_response>
</example>
Luôn dùng Calculator để tính, không tự tính thủ công.""")

    user_query = "Tính căn bậc hai của 16 cộng với 4 chia 2"

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    ]

    while True:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=[system_instruction],
            ),
        )

        model_reply = response.text
        print("Model:", model_reply)

        contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=model_reply)])
        )

        if "Answer:" in model_reply:
            break

        expr = extract_action(model_reply)
        if expr is None:
            print("Không tìm thấy hành động calculator.")
            break

        observation = run_calculator(expr)
        print("Observation:", observation)

        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=f"Observation: {observation}")])
        )


if __name__ == "__main__":
    generate()