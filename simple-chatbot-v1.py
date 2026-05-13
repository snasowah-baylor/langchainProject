import os
from dotenv import load_dotenv  # type: ignore
from langchain.agents import create_agent  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful chat assistant. Be clear, concise and polite. Understand "
    "the user's intent and respond directly. Stay professional and safe. and provide"
    "a helpful answer. If you don't know the answer, say you don't know. Stay professional and safe.",
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Explain machine learning in short."}],
    }
)

print(result["messages"][1].content)
