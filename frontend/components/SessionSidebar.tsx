"use client";

import type { AppConfig, SessionSummary } from "@/lib/types";
import ProviderBadge from "./ProviderBadge";

interface Props {
  sessions: SessionSummary[];
  activeId: string | null;
  config: AppConfig | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

export default function SessionSidebar({
  sessions,
  activeId,
  config,
  open,
  onSelect,
  onNew,
  onDelete,
  onClose,
}: Props) {
  return (
    <>
      {/* Scrim, mobile only — the sidebar is a drawer below the lg breakpoint. */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-ink/30 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <nav
        aria-label="Chat sessions"
        className={`fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-gray-200 bg-surface transition-transform lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="border-b border-gray-200 p-4">
          <h1 className="text-sm font-semibold text-ink">The Lenny Growth Assistant</h1>
          <div className="mt-2">
            <ProviderBadge config={config} />
          </div>
        </div>

        <div className="p-3">
          <button
            type="button"
            onClick={onNew}
            className="w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-700"
          >
            + New chat
          </button>
        </div>

        <ul className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
          {sessions.length === 0 && (
            <li className="px-3 py-6 text-center text-xs text-ink-muted">
              No conversations yet.
            </li>
          )}
          {sessions.map((s) => (
            <li key={s.id}>
              <div
                className={`group flex items-center gap-1 rounded-lg transition ${
                  s.id === activeId ? "bg-accent-soft" : "hover:bg-surface-raised"
                }`}
              >
                <button
                  type="button"
                  onClick={() => onSelect(s.id)}
                  aria-current={s.id === activeId ? "page" : undefined}
                  className="min-w-0 flex-1 px-3 py-2 text-left"
                >
                  <span
                    className={`block truncate text-sm ${
                      s.id === activeId ? "font-medium text-accent" : "text-ink-soft"
                    }`}
                  >
                    {s.title}
                  </span>
                  <span className="block text-[11px] text-ink-muted">
                    {s.message_count} message{s.message_count === 1 ? "" : "s"}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(s.id)}
                  aria-label={`Delete conversation: ${s.title}`}
                  className="mr-1 rounded p-1.5 text-ink-muted opacity-0 transition hover:bg-red-50 hover:text-red-600 focus-visible:opacity-100 group-hover:opacity-100"
                >
                  &#10005;
                </button>
              </div>
            </li>
          ))}
        </ul>

        {config && (
          <footer className="border-t border-gray-200 px-4 py-3 text-[11px] leading-snug text-ink-muted">
            {config.knowledge_base_ready ? (
              <>
                {config.episodes} episodes · {config.chunks.toLocaleString()} passages indexed
              </>
            ) : (
              <>Knowledge base is still building...</>
            )}
          </footer>
        )}
      </nav>
    </>
  );
}
