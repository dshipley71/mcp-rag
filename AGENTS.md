# AGENTS.md

## Project
Deterministic MCP-RAG v1

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

## Implementation Rules
- Prefer simple Python modules over complex abstractions.
- Use explicit functions and data models.
- Keep configuration in YAML files.
- Keep orchestration logic readable and testable.
- Every pipeline stage must map clearly to a role in `mcp_catalog.yaml`.
- Every routing decision must map clearly to `routing_rules.yaml`.

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

## Current v1 Pipeline
1. bm25_search
2. vector_search
3. document_fetch
4. rerank
5. answer

## Current v1 Constraints
- hybrid retrieval always on
- max retries = 1
- must use context only
- must cite sources
- stop if no evidence

## Developer Notes
This project is being built step by step.
Favor clarity over completeness.
Favor determinism over flexibility.
Favor explicit behavior over smart behavior.