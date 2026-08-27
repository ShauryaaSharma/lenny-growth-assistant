# 02 — Concurrent downloads starved each other, and a false "success"

**Context:** session 1 ended with the Ollama model pull (~2GB) and the Docker
backend build both queued as background tasks, running at the same time, to
use idle wall-clock time efficiently.

## What went wrong

At the end of session 1, the agent reported to the user: *"Docker backend
image built"* — based on a background task notification whose summary line
said the command "completed (exit code 0)." Session 2 opened by verifying that
claim before trusting it, rather than proceeding as if the backend were ready.

```
docker images 2>/dev/null | grep -iE "lenny|python:3.12" || echo "NO backend image"
→ NO backend image

tail of the actual build log:
   #9 >>> RUN apt-get update && apt-get install -y --no-install-recommends git curl ...
   failed to solve: process ... did not complete successfully: exit code: 100
```

The build had genuinely failed. The background-task summary line reporting
"completed" referred to the shell command exiting (it did — with a non-zero
Docker Compose exit that got reported at the wrong layer), not to the build
having succeeded. Root cause: the Ollama pull and the `apt-get` step inside the
Docker build were both pulling over the same constrained network link at the
same time, and `apt-get update` timed out.

## The correction

1. Never re-trust a "completed" status on a long-running background command
   without an independent check (`docker images`, in this case) — this is now
   a standing practice for the rest of the build, not just this one incident.
2. Serialized the two downloads: let the Ollama model pull finish alone first,
   confirmed it with `ollama list` actually showing the model, *then* started
   the Docker build alone.
3. Re-ran the build; it completed cleanly this time, verified again by
   `docker images` actually listing the resulting image, not just by the exit
   code of the command that triggered it.

## Why this belongs in the log

This is exactly the kind of failure the brief's "include failed attempts and
how you corrected them" instruction is asking for: a background process
reported success while having actually failed, and the fix was procedural
(verify state independently of a status message) rather than a code change.
