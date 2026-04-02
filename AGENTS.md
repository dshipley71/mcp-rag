# AGENTS.md

## Project
Deterministic MCP-RAG v2

## Purpose
Build a retrieval-augmented generation system that uses MCP-style tool roles defined in `mcp_catalog.yaml` and routing behavior defined in `routing_rules.yaml`.

The system must behave predictably and avoid unnecessary variation between runs.

## Source of Truth
The agent must treat these files as authoritative:
- `mcp_catalog.yaml`
- `routing_rules.yaml`

Do not invent new MCP roles, pipeline stages, or routing behavior unless explicitly requested.

## Build Rules
- Keep the implementation minimal and deterministic.
- Do not add features outside the current step.
- Do not introduce optional frameworks unless required.
- Do not rename roles defined in `mcp_catalog.yaml`.
- Do not change the order of the pipeline defined in `routing_rules.yaml`.
- Do not add more than one retry loop.
- Do not add dynamic tool discovery.
- Do not add runtime MCP registry browsing.
- Do not add external web calls.

## MCP Server Bindings
The approved external components are:
- `modelcontextprotocol/filesystem` for source-of-truth document access
- `Unstructured-IO/UNS-MCP` for multi-format parsing and normalization
- `haseebkhalid1507/velocirag` for retrieval and retrieval-side ordering
- `jonigl/ollama-mcp-bridge` for answer generation

### Ollama Rule
- All answer generation must go through Ollama MCP Bridge.
- Ollama MCP Bridge must be configured to use Ollama Cloud.
- Do not use local Ollama models for answer generation.
- Do not replace Ollama MCP Bridge with a direct LLM provider call unless explicitly requested.

## Implementation Rules
- Prefer simple Python modules over complex abstractions.
- Use explicit functions and data models.
- Keep configuration in YAML files.
- Keep orchestration logic readable and testable.
- Every pipeline stage must map clearly to a role in `mcp_catalog.yaml`.
- Every routing decision must map clearly to `routing_rules.yaml`.
- Use direct MCP calls for retrieval.
- Use the bridge HTTP API for answer generation.

## Answering Rules
- The system must only answer from retrieved evidence.
- The system must cite the retrieved source chunks it used.
- The system must stop if evidence is insufficient.
- The system must not hallucinate sources or claims.

## File Change Rules
- Modify only the files needed for the current step.
- Do not create extra files unless necessary.
- Keep names stable and obvious.
- Output complete files when making changes.

## Current v2 Pipeline
1. bm25_search
2. vector_search
3. document_fetch
4. rerank
5. answer

## Current v2 Constraints
- hybrid retrieval always on
- max retries = 1
- must use context only
- must cite sources
- stop if no evidence
- answer generation must use Ollama Cloud through Ollama MCP Bridge

## Developer Notes
This project is being built step by step.
Favor clarity over completeness.
Favor determinism over flexibility.
Favor explicit behavior over smart behavior.
