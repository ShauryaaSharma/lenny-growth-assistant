"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation, Message } from "@/lib/types";

/**
 * A single turn.
 *
 * Assistant messages carry their citations inline. The design decision here:
 * citations are collapsed by default and expand on click. Showing eight source
 * cards under every answer buries the answer itself, but hiding sources
 * entirely defeats the point of a grounded assistant — so the count is always
 * visible and the detail is one click away.
 */

function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (citations.length === 0) return null;

  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-center gap-1.5 text-xs font-medium text-ink-muted transition hover:text-ink"
      >
        <span aria-hidden="true" className={`transition-transform ${open ? "rotate-90" : ""}`}>
          &#9656;
        </span>
        {citations.length} source{citations.length === 1 ? "" : "s"} from Lenny&apos;s Podcast
      </button>

      {open && (
        <ol className="mt-2 space-y-1.5">
          {citations.map((c, i) => (
            <li key={c.chunk_id} className="flex gap-2 text-xs leading-snug">
              <span className="shrink-0 font-mono text-ink-muted">[{i + 1}]</span>
              <span className="min-w-0">
                {c.url ? (
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-accent underline underline-offset-2"
                  >
                    {c.episode_title}
                  </a>
                ) : (
                  <span className="font-medium text-ink">{c.episode_title}</span>
                )}
                <span className="text-ink-muted">
                  {" · "}
                  {c.guest}
                  {c.timestamp && ` · ${c.timestamp}`}
                </span>
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

interface Props {
  message: Message;
  onOpenArtifact?: (id: string) => void;
  artifactIds?: string[];
  artifactTitles?: Record<string, string>;
}

export default function MessageBubble({
  message,
  onOpenArtifact,
  artifactIds = [],
  artifactTitles = {},
}: Props) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-2.5 text-[15px] leading-relaxed text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-surface px-4 py-3 shadow-sm ring-1 ring-gray-100">
        <div className="prose-chat">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>

        {artifactIds.length > 0 && onOpenArtifact && (
          <div className="mt-3 flex flex-wrap gap-2">
            {artifactIds.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => onOpenArtifact(id)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 bg-surface-sunken px-3 py-1.5 text-xs font-medium text-ink transition hover:border-accent-ring hover:bg-accent-soft"
              >
                <span aria-hidden="true">&#128196;</span>
                {artifactTitles[id] || "Open artifact"}
              </button>
            ))}
          </div>
        )}

        <CitationList citations={message.citations} />

        {(message.model || message.latency_ms) && (
          <p className="mt-2 text-[11px] text-ink-muted">
            {message.model}
            {message.latency_ms ? ` · ${(message.latency_ms / 1000).toFixed(1)}s` : ""}
          </p>
        )}
      </div>
    </div>
  );
}
