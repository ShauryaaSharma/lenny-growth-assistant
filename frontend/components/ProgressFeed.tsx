"use client";

import type { ProgressEvent } from "@/lib/types";

/**
 * What the agent is doing while a turn runs: each search with what it found,
 * each document it creates, and each time a grounding guard steps in. A
 * local model can take a minute or more per turn; this is the difference
 * between watching it work and wondering whether it hung.
 */

export interface ProgressStep {
  id: number;
  label: string;
  status: "running" | "done" | "failed";
  summary?: string;
  tool?: string;
}

const TOOL_LABELS: Record<string, (detail: string) => string> = {
  search_transcripts: (d) => (d ? `Searching transcripts for “${d}”` : "Searching transcripts"),
  write_ship30_essay: (d) => (d ? `Writing a Ship 30 essay on “${d}”` : "Writing a Ship 30 essay"),
  create_artifact: (d) => (d ? `Creating “${d}”` : "Creating a document"),
};

// Only the guards a user would want to know about. A blocked duplicate
// artifact is housekeeping, not news.
const GUARD_LABELS: Partial<Record<string, string>> = {
  forced_retrieval: "Checking the transcripts before answering",
  ungrounded: "Nothing relevant found — making sure the answer says so",
  ungrounded_artifact: "Searching the transcripts before writing the document",
  artifact_nudge: "Putting that into a document",
};

/** Fold one event into the step list. Pure, so it is safe in a setState updater. */
export function applyProgress(steps: ProgressStep[], event: ProgressEvent): ProgressStep[] {
  const nextId = steps.length ? steps[steps.length - 1].id + 1 : 0;
  // Anything that happens means the model call before it has finished.
  const settled = steps.map((s) =>
    s.status === "running" && !s.tool ? { ...s, status: "done" as const } : s,
  );

  switch (event.type) {
    case "thinking": {
      const last = settled[settled.length - 1];
      if (last && !last.tool && last.label === "Thinking") return steps;
      const label = settled.length === 0 ? "Reading your question" : "Thinking";
      return [...settled, { id: nextId, label, status: "running" }];
    }
    case "tool_start": {
      const describe = TOOL_LABELS[event.tool];
      const label = describe ? describe(event.detail) : `Running ${event.tool}`;
      return [...settled, { id: nextId, label, status: "running", tool: event.tool }];
    }
    case "tool_end": {
      const index = settled.map((s) => s.tool === event.tool && s.status === "running").lastIndexOf(true);
      if (index === -1) return settled;
      const updated = [...settled];
      updated[index] = {
        ...updated[index],
        status: event.ok ? "done" : "failed",
        summary: event.summary,
      };
      return updated;
    }
    case "guard": {
      const label = GUARD_LABELS[event.guard];
      return label ? [...settled, { id: nextId, label, status: "done" }] : settled;
    }
  }
}

export default function ProgressFeed({ steps }: { steps: ProgressStep[] }) {
  if (steps.length === 0) return null;
  return (
    <ol className="mt-2 space-y-1 text-xs text-ink-muted" aria-label="What the assistant is doing">
      {steps.map((s) => (
        <li key={s.id} className="flex items-start gap-2">
          <span aria-hidden className="mt-px w-3 shrink-0 text-center">
            {s.status === "running" ? "…" : s.status === "failed" ? "✕" : "✓"}
          </span>
          <span>
            {s.label}
            {s.summary && <span className="text-ink-soft"> &mdash; {s.summary}</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}
