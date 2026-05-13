import os
from dotenv import load_dotenv  # type: ignore
from langchain.agents import create_agent  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=2)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful chat assistant. Be clear, concise and polite. Understand the user's intent and respond directly. Stay professional and safe. "
    " and provide a helpful answer. If you don't know the answer, say you don't know. Stay professional and safe.",
)

chat_history = []
print("\n\n")
print(
    "Hi, I'm a Chat Assistant. My name is Steve. I'm ready! type 'bye' or 'exit' to end the conversation."
)

while True:
    print("\n\n")
    user_input = input("You: ").strip()
    print("\n")
    if user_input.lower() in ["bye", "exit"]:
        print("Assistant 🤖: Goodbye! 👋")
        break

    messages = chat_history + [{"role": "user", "content": user_input}]
    result = agent.invoke({"messages": messages})

    """ rint(result) """

    try:
        reply = result["messages"][-1].content
    except Exception as e:
        reply = str(e)

    print(f"Steve 🤖: {reply} \n")
    print("-" * 60)

    # update chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})
