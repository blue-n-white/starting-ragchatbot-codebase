# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Python/FastAPI backend with ChromaDB vector search and Claude API, vanilla JS frontend.

## Important Rules

Always use `uv` to install dependencies and run Python code. Never use `pip`.

Always add and commit changes from this working directory and its subdirectories, and push to `origin` (`github.com:blue-n-white/starting-ragchatbot-codebase`).

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the application (starts on http://localhost:8000)
cd backend && uv run uvicorn app:app --reload --port 8000

# Or use the launch script
./run.sh
```

Swagger docs available at `http://localhost:8000/docs`.

No test suite or linter is configured.

## Environment Setup

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

**Request flow:** Frontend → `POST /api/query` → `RAGSystem.process_query()` → Claude (with tool calling) → `search_course_content` tool → ChromaDB vector search → Claude generates answer with retrieved context → response with sources returned to frontend.

**Backend (`backend/`):**

- `app.py` — FastAPI entry point. Defines API routes (`/api/query`, `/api/courses`), serves frontend as static files, loads course documents on startup.
- `rag_system.py` — Orchestrator. Coordinates document processing, vector storage, and AI generation. This is the central module that ties everything together.
- `vector_store.py` — ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunked text). Uses `all-MiniLM-L6-v2` embeddings via sentence-transformers.
- `document_processor.py` — Parses course text files into structured data (course title, lessons, links) and chunks content (800 chars, 100 char overlap).
- `ai_generator.py` — Claude API integration. Sends queries with tool definitions, executes tool calls in a loop, returns final response. Uses zero temperature.
- `search_tools.py` — Defines the `search_course_content` tool schema and `ToolManager` that dispatches tool calls to vector store searches.
- `session_manager.py` — Tracks per-user conversation history (default limit: 2 messages). History is injected into the system prompt.
- `config.py` — Loads env vars and sets defaults (model, chunk size, collection names).
- `models.py` — Pydantic models for API request/response schemas.

**Frontend (`frontend/`):** Single-page vanilla JS app. No build step. Uses marked.js for markdown rendering. Dark theme with sidebar showing course stats and suggested questions.

**Data (`docs/`):** Course text files in a specific format — each file starts with course metadata (title, link, instructor) followed by numbered lessons with content. These are auto-loaded on startup.

## Course Document Format

```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[content...]
```

## Key Dependencies

- `chromadb` 1.0.15 — vector database
- `anthropic` 0.58.2 — Claude API client
- `sentence-transformers` 5.0.0 — embedding model
- `fastapi` 0.116.1 — web framework
- Python 3.13 required
