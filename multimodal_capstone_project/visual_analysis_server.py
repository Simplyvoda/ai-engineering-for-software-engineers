import os
import mimetypes
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

# The MCP client doesn't pass its environment to this subprocess, so load .env here
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize the FastMCP server
mcp = FastMCP("VisualAnalysisServer")

@mcp.tool()
def get_image_description(file_path: str) -> str:
    """
    Loads an image from a server-accessible file path and returns a detailed,
    descriptive paragraph about its content. If the image is of a known landmark,
    it will be specifically identified. This description is intended to be used
    as a high-quality search query for a research tool.

    Args:
        file_path: The absolute path to the image file, which must be
                   accessible by the server running this tool.

    Returns:
        A single string containing a detailed description of the image.
        Returns an error message if analysis fails.
    """
    try:
        image_path = Path(file_path)
        if not image_path.is_file():
            return f"Error: file not found at path: {file_path}"

        # Guess the MIME type from the file extension
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "application/octet-stream"  # A generic default

        # The image bytes go straight to Gemini from here, so the agent's model
        # only ever handles the short file path, never the image data itself
        image_part = types.Part.from_bytes(
            mime_type=mime_type,
            data=image_path.read_bytes()
        )
        
        # --- prompt text ---
        prompt_text = (
            "Analyze this image in detail. Provide a concise, one-paragraph description. "
            "If it is a famous landmark, work of art, or specific location, identify it by name. "
            "Focus on the most important and defining elements in the image that would be useful for a web search. "
            "For example, instead of 'a building', say 'the Eiffel Tower in Paris'. "
            "Do not add any conversational filler; return only the description."
        )

        client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
        # The generate_content call now uses the correctly typed image_part object
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=[image_part, prompt_text],
        )
        
        # --- Return the single description string directly ---
        # The .strip() removes any leading/trailing newlines or spaces from the model's output
        description = response.text.strip()
        
        return description

    except Exception as e:
        return f"Error analyzing image: {e}"
        
if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run(transport="stdio")