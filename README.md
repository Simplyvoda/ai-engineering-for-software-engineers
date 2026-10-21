# AI Engineering for Software Engineers

My hands-on work from Educative's [AI Skills Learning Path for Software Engineers](https://www.educative.io/path/learning-ai-skills-for-software-engineers). The path goes from using AI tools to building production AI systems. It covers how LLMs work, prompt engineering, tool use with the Model Context Protocol (MCP), agents, retrieval-augmented generation (RAG), multi-agent orchestration, and shipping AI to production.

Each folder is a self-contained project built while working through the path.

## Projects

| Project | What it does | Concepts |
| --- | --- | --- |
| [`llm-cli-tool/`](llm-cli-tool/) | Command-line tool that sends a prompt to an LLM and streams the reply, then reports tokens, latency and cost | Tokens, model selection, streaming, cost tracking |
| [`prompt_eval_pipeline/`](prompt_eval_pipeline/) | Compares system prompts for a rental-review sentiment classifier, tracks accuracy across runs, and A/B tests prompts with a significance check | Prompt engineering, chain-of-thought, automated evaluation, A/B testing |
| [`server_mcp/`](server_mcp/) | Two MCP servers (weather and tasks) and LangGraph clients that connect to one or many servers and use their tools, prompts and resources | MCP tools, prompts and resources, multi-server clients, agents |
| [`multimodal_capstone_project/`](multimodal_capstone_project/) | Image Research Assistant: a web app where you upload an image and ask a question. A LangGraph agent chains two MCP servers, one that describes the image with Gemini and one that researches it on Wikipedia | Multimodal input, tool chaining across MCP servers, agents, Gradio UI |
| [`ai_agent/`](ai_agent/) | Three agents built with Google's Agent Development Kit (ADK): a greeter with structured input and output, a research assistant that writes a report from Wikipedia, arXiv and Google Search, and a multi-agent version where a controller delegates to specialist agents | ADK agents, tools, structured output, multi-agent orchestration, agents as tools |

The loose scripts at the top level are short examples of patterns for calling LLMs from production code:

| Script | Pattern |
| --- | --- |
| `streaming.py` | Streaming a response token by token |
| `retry_logic.py` | Retrying on rate limits with exponential backoff and jitter |
| `tenacity_retry_logic..py` | The same retry policy, using the `tenacity` library |
| `json_logger.py` | Structured JSON logging of model, tokens and latency for every call |
| `pydantic_models.py` | Structured output parsed into Pydantic models |

## Setup

Each project has its own virtual environment, and all projects share one `.env` file for API keys.

**1. Add your API keys.** Copy the example file in this folder and fill in your own keys:

```bash
cp .env.example .env
```

| Key | Used by | Get one at |
| --- | --- | --- |
| `OPENAI_API_KEY` | `llm-cli-tool`, `prompt_eval_pipeline`, top-level scripts | https://platform.openai.com/api-keys |
| `GOOGLE_GEMINI_API_KEY` | `server_mcp`, `multimodal_capstone_project` | https://aistudio.google.com/apikey |
| `OPENWEATHERMAP_API_KEY` | `server_mcp` weather server | https://home.openweathermap.org/api_keys |
| `GEMINI_MODEL` | `server_mcp`, `multimodal_capstone_project` | For example `gemini-3.5-flash` |
| `GOOGLE_API_KEY` | `ai_agent` (ADK reads this name) | The same key as `GOOGLE_GEMINI_API_KEY` |
| `ADK_MODEL` | `ai_agent` research agents | For example `gemini-3.5-flash-lite` |

`.env` is listed in `.gitignore`, so your keys stay on your machine. The MCP projects, `llm-cli-tool` and the ADK agents load the file automatically. `prompt_eval_pipeline` and the top-level scripts read `OPENAI_API_KEY` from the shell environment, so export it before running them:

```bash
export OPENAI_API_KEY="your-key"
```

**2. Create a virtual environment for the project you want to run.** For example:

```bash
cd multimodal_capstone_project
python3 -m venv .venv
source .venv/bin/activate
```

If you rename or move a project folder, delete its `.venv` and create it again, because virtual environments store their own path.

**3. Install the project's packages:**

| Project | Install command |
| --- | --- |
| `llm-cli-tool` | `pip install click openai python-dotenv tiktoken tenacity` |
| `prompt_eval_pipeline` | `pip install -r requirements.txt` |
| `server_mcp` | `pip install python-dotenv mcp langgraph langchain-core langchain-google-genai langchain-mcp-adapters requests` |
| `multimodal_capstone_project` | `pip install python-dotenv mcp langgraph langchain-core langchain-google-genai langchain-mcp-adapters google-genai wikipedia gradio` |
| `ai_agent` | `pip install google-adk wikipedia arxiv` |

The three agents in `ai_agent/` share one virtual environment, `ai_agent/.venv`. Activate that one before running `adk`, not the one in this folder.

## Running

Run each project from inside its own folder, because the scripts open data files with relative paths.

```bash
# LLM CLI: stream a reply and report cost
cd llm-cli-tool
python cli.py "Explain context windows in two sentences" --model gpt-4o-mini

# Prompt evaluation: score every prompt in prompts/, then chart and A/B test
cd prompt_eval_pipeline
python evaluation_system.py
python analyze_result.py
python visualize.py
python ab_test.py

# MCP: chat with an agent connected to the weather and task servers
cd server_mcp
python multi_mcp_client.py

# Multimodal: web app for researching an uploaded image
cd multimodal_capstone_project
python mcp_client.py              # then open http://127.0.0.1:7860

# ADK agents: run from ai_agent/, not from inside an agent's folder
cd ai_agent
adk web                           # browser UI: pick an agent from the list
adk run multi_agent_research_assistant   # or chat in the terminal
```

For the research agents, type a topic such as `Research CRISPR gene editing`. They save the report as a text file in `ai_agent/`. Running `python agent.py` on its own only defines the agent and exits.

Gemini counts Google Search requests separately from model requests, and my API key has no search quota on any model, so search requests fail with `429 RESOURCE_EXHAUSTED`. The multi-agent research assistant has its `web_searcher` agent switched off for that reason and researches with Wikipedia and arXiv only. The single-agent `research_assistant` still uses Google Search, so it needs a key with search quota.

In `multi_mcp_client.py`, `/resources` lists the resources each server provides, and `/resource <server> <uri>` loads one into the conversation, for example `/resource tasks file://meeting_notes`. The `/prompts` and `/prompt` commands are currently commented out in that client.

## Path modules

1. How LLMs Actually Work
2. Prompt Engineering That Works
3. Agentic Coding Workflows
4. Extending AI with Tools and MCP
5. Building Your First AI Agent
6. RAG and Memory Systems
7. Multi-Agent Orchestration
8. Shipping AI to Production
