# 00 — Planning and corpus selection

**Context:** first message of the session. The user shared the take-home brief
as a `.docx` and asked for a plan before any code was written.

## What the agent did

1. Extracted and read the full brief from the `.docx` (via a zipfile/XML read,
   since no dedicated tool was loaded yet).
2. Identified the brief's evaluation weighting: only 2 of 7 rubric categories
   are pure code quality; the rest are discovery framing, agentic architecture,
   operability, and communication. Concluded that documentation and tests were
   not the "buffer" to cut if time got tight.
3. Flagged a real tension in the brief before writing any code: it names the
   Claude Agent SDK / Pi Coding Agent as the suggested agent layer, but also
   *requires* the submitted demo to run on local Ollama. Neither SDK executes
   against Ollama. This was surfaced to the user as the single highest-value
   thing to get right, rather than discovered mid-build.
4. Searched for the actual transcript corpus rather than assuming one, and
   compared three candidate repositories on frontmatter richness (which one
   would support real timestamped citations) before recommending
   `ChatPRD/lennys-podcast-transcripts`.
5. Asked the user three clarifying questions via `AskUserQuestion` rather than
   guessing: hardware specs (16GB, no GPU — this fixed the model tier for the
   rest of the build), cloud-provider access (no Anthropic/OpenAI key — this
   is what forced the Hugging Face router adapter design), and whether to find
   the transcript source (user delegated this).

## Why this order mattered

Answering "what hardware" and "what API access" before writing any backend
code meant the `LLMProvider` abstraction, the embedding model choice
(`fastembed`/ONNX instead of `sentence-transformers`/torch), and the default
Ollama model (`llama3.2:3b`) were all sized correctly from the first line of
code, rather than retrofitted later.

## Correction made mid-plan

The initial verbal plan suggested Supabase/Railway as the primary database.
On reflection (documented directly to the user), local Postgres in Compose was
chosen instead specifically because it gives a genuinely reproducible
one-command start with zero evaluator signup — Supabase/Railway remain
supported via `DATABASE_URL`, but are not the shipped default. This is recorded
as an explicit scope decision in `docs/PRD.md`, not a silent choice.
