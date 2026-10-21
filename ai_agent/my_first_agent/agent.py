from google.adk.agents.llm_agent import Agent
from pydantic import BaseModel, Field
from google.adk.planners import BuiltInPlanner
from google.genai import types


class GreetingRequest(BaseModel):
    """Input schema for specifying the language of the greeting."""

    language: str = Field(description="The language to greet the user in.")


class GreetingResponse(BaseModel):
    """Output schema for the structured greeting response."""

    greeting_message: str = Field(description="The final, formatted greeting.")


def greeting_tool() -> str:
    """Returns a warm, friendly greeting."""
    return "Hello from your specialized greeting tool! Welcome."


def create_greeting(name: str, language: str = "English") -> str:
    """Creates a personalized greeting for a user in a specified language.

    Args:
        name (str): The name of the person to greet.
        language (str): The language for the greeting. Defaults to English.
    """
    if language.lower() == "spanish":
        return f"Hola, {name}! Cómo estás?"
    else:
        return f"Hello, {name}! How are you?"


root_agent = Agent(
    model="gemini-3.5-flash",
    name="root_agent",
    description="A friendly agent that provides a special greeting.",
    # instruction="You are a friendly agent. When the user greets you, you MUST use the greeting_tool to respond.",
    instruction="""You are a helpful greeter. When the user asks for a greeting,
    use the `create_greeting` tool to generate it.""",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    # tools=[greeting_tool],
    tools=[create_greeting],
    input_schema=GreetingRequest,
    output_schema=GreetingResponse,
    output_key="final_greeting",
)
