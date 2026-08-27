"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Artifact } from "@/lib/types";

/**
 * Artifact viewer — layer 2 of the artifact security model.
 *
 * The server already sanitised this content (app/security/sanitize.py). This
 * layer assumes that sanitiser could fail and contains the damage anyway.
 *
 * HTML is rendered in an iframe with `sandbox="allow-scripts"` and deliberately
 * WITHOUT `allow-same-origin`. That combination puts the document on an opaque
 * origin, so even if script executes it cannot:
 *   - read or write the parent DOM
 *   - read our cookies, localStorage, or sessionStorage
 *   - make same-origin requests to our API
 *
 * A CSP of `default-src 'none'` is injected into the document head, which blocks
 * every network request the page might attempt — so a successful injection has
 * no channel to exfiltrate what it sees.
 *
 * Scripts are allowed at all only so that legitimate interactive artifacts
 * (a chart, a toggle, a calculator) work. Everything they could do harm with is
 * removed by the two constraints above.
 *
 * Markdown never touches `dangerouslySetInnerHTML`: react-markdown parses to a
 * React tree and, with no rehype-raw plugin, embedded HTML is rendered as text.
 */

const CSP = [
  "default-src 'none'",
  "style-src 'unsafe-inline'",
  "img-src data:",
  "font-src data:",
  "script-src 'unsafe-inline'",
  "form-action 'none'",
  "base-uri 'none'",
].join("; ");

const IFRAME_SANDBOX = "allow-scripts";

function buildSrcDoc(html: string): string {
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="${CSP}">
<style>
  :root { color-scheme: light; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #111827;
    background: #ffffff;
    padding: 24px;
    line-height: 1.6;
  }
  img, table { max-width: 100%; }
  table { border-collapse: collapse; }
  td, th { border: 1px solid #e5e7eb; padding: 6px 10px; }
  pre { overflow-x: auto; }
</style>
</head>
<body>
${html}
</body>
</html>`;
}

interface Props {
  artifact: Artifact | null;
  onClose: () => void;
}

export default function ArtifactViewer({ artifact, onClose }: Props) {
  const [tab, setTab] = useState<"preview" | "source">("preview");
  const [showSecurity, setShowSecurity] = useState(false);

  const srcDoc = useMemo(
    () => (artifact?.kind === "html" ? buildSrcDoc(artifact.content) : ""),
    [artifact],
  );

  if (!artifact) {
    return (
      <aside className="flex h-full flex-col items-center justify-center gap-2 border-l border-gray-200 bg-surface-sunken p-8 text-center">
        <div className="text-3xl" aria-hidden="true">
          &#128196;
        </div>
        <h2 className="text-sm font-semibold text-ink">No artifact yet</h2>
        <p className="max-w-xs text-sm text-ink-muted">
          Ask for a document, a one-pager, a table, or an essay and it will render here
          beside the chat.
        </p>
      </aside>
    );
  }

  const findings = artifact.sanitizer_report?.findings ?? [];

  return (
    <aside
      className="flex h-full flex-col border-l border-gray-200 bg-surface"
      aria-label="Artifact viewer"
    >
      <header className="flex items-center gap-2 border-b border-gray-200 px-4 py-3">
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-sm font-semibold text-ink" title={artifact.title}>
            {artifact.title}
          </h2>
          <p className="text-xs text-ink-muted">
            {artifact.kind === "html" ? "HTML / CSS" : "Markdown"}
            {findings.length > 0 && (
              <>
                {" · "}
                <button
                  type="button"
                  onClick={() => setShowSecurity((v) => !v)}
                  className="font-medium text-amber-700 underline underline-offset-2"
                >
                  {findings.length} item{findings.length === 1 ? "" : "s"} removed
                </button>
              </>
            )}
          </p>
        </div>

        <div
          className="flex rounded-md border border-gray-200 p-0.5"
          role="tablist"
          aria-label="Artifact view mode"
        >
          {(["preview", "source"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`rounded px-2.5 py-1 text-xs font-medium capitalize transition ${
                tab === t
                  ? "bg-accent-soft text-accent"
                  : "text-ink-muted hover:text-ink"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={onClose}
          aria-label="Close artifact viewer"
          className="rounded p-1.5 text-ink-muted transition hover:bg-surface-raised hover:text-ink"
        >
          &#10005;
        </button>
      </header>

      {showSecurity && findings.length > 0 && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-900">
          <p className="font-semibold">Content removed before rendering</p>
          <ul className="mt-1 list-inside list-disc">
            {findings.map((f) => (
              <li key={f}>{f.replace(/_/g, " ")}</li>
            ))}
          </ul>
          <p className="mt-2">
            Generated HTML is treated as untrusted. It is sanitised server-side and
            rendered in a sandboxed frame with no same-origin access and no network
            permission.
          </p>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto">
        {tab === "source" ? (
          <pre className="whitespace-pre-wrap break-words p-4 font-mono text-xs leading-relaxed text-ink-soft">
            {artifact.content}
          </pre>
        ) : artifact.kind === "html" ? (
          <iframe
            title={artifact.title}
            sandbox={IFRAME_SANDBOX}
            srcDoc={srcDoc}
            className="h-full w-full border-0"
            // referrerPolicy is belt-and-braces: the CSP already blocks egress.
            referrerPolicy="no-referrer"
          />
        ) : (
          <div className="prose-artifact p-6">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{artifact.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </aside>
  );
}
