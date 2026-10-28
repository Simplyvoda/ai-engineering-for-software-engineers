"""Run the weather agent on a few example questions.

Usage:
    python main.py                       # runs the built-in examples
    python main.py "Weather in Lagos?"   # ask your own question
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from weather_agent import build_graph

EXAMPLES = [
    "What's the weather in Paris and London right now?",
    "How's the weather in Lagos? Use Fahrenheit please.",
    "What's the weather in Atlantisville?",  # city that doesn't exist
    "Thanks!",  # no tool needed
]


def run(app, question: str) -> None:
    print("=" * 60)
    print(f"USER: {question}")
    print("-" * 60)
    # recursion_limit stops a runaway llm<->tools loop.
    result = app.invoke({"messages": [HumanMessage(content=question)]}, {"recursion_limit": 12})
    for msg in result["messages"][1:]:
        if getattr(msg, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in msg.tool_calls)
            print(f"[llm -> tools] {calls}")
        elif msg.type == "tool":
            print(f"[tool:{msg.name}] {msg.content}")
    print(f"\nASSISTANT: {result['messages'][-1].content}\n")


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    app = build_graph(llm)

    questions = sys.argv[1:] or EXAMPLES
    for q in questions:
        run(app, q)


if __name__ == "__main__":
    main()
