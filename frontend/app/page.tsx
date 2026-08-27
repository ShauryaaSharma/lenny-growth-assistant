"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ArtifactViewer from "@/components/ArtifactViewer";
import Composer from "@/components/Composer";
import MessageBubble from "@/components/MessageBubble";
import SessionSidebar from "@/components/SessionSidebar";
import { ApiError, api } from "@/lib/api";
import type {
  AppConfig,
  Artifact,
  Message,
  SessionSummary,
} from "@/lib/types";

/** Poll interval while the knowledge base is still seeding on first boot. */
const KB_POLL_MS = 4000;

export default function Home() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [artifacts, setArtifacts] = useState<Record<string, Artifact>>({});
  const [messageArtifacts, setMessageArtifacts] = useState<Record<string, string[]>>({});
  const [openArtifactId, setOpenArtifactId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [booting, setBooting] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshConfig = useCallback(async () => {
    try {
      setConfig(await api.getConfig());
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    }
  }, []);

  // Initial boot: config + sessions, creating one if the user has none.
  useEffect(() => {
    (async () => {
      await refreshConfig();
      try {
        const list = await api.listSessions();
        setSessions(list);
        if (list.length > 0) {
          setActiveId(list[0].id);
        } else {
          const created = await api.createSession();
          setSessions([created]);
          setActiveId(created.id);
        }
      } catch (e) {
        if (e instanceof ApiError) setError(e);
      } finally {
        setBooting(false);
      }
    })();
  }, [refreshConfig]);

  // Keep polling until the corpus finishes seeding, so the UI un-blocks itself
  // without the user needing to reload.
  useEffect(() => {
    if (!config || config.knowledge_base_ready) return;
    const t = setInterval(refreshConfig, KB_POLL_MS);
    return () => clearInterval(t);
  }, [config, refreshConfig]);

  // Load the selected conversation.
  useEffect(() => {
    if (!activeId) return;
    (async () => {
      try {
        const detail = await api.getSession(activeId);
        setMessages(detail.messages);
        setOpenArtifactId(null);
        setMessageArtifacts({});
        // Artifact bodies are fetched lazily; the list gives us titles only.
        const summaries: Record<string, Artifact> = {};
        for (const a of detail.artifacts) {
          summaries[a.id] = { ...a, content: "", sanitizer_report: null };
        }
        setArtifacts(summaries);
      } catch (e) {
        if (e instanceof ApiError) setError(e);
      }
    })();
  }, [activeId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function handleSend(text: string) {
    if (!activeId) return;
    setError(null);
    setBusy(true);

    // Optimistic user turn so the UI responds instantly even when a local model
    // takes 30+ seconds to answer.
    const optimistic: Message = {
      id: `pending-${Date.now()}`,
      role: "user",
      content: text,
      provider: null,
      model: null,
      citations: [],
      latency_ms: null,
      created_at: new Date().toISOString(),
    };
    setMessages((m) => [...m, optimistic]);

    try {
      const res = await api.sendMessage(activeId, text);
      setMessages((m) => [...m, res.message]);

      if (res.artifacts.length > 0) {
        setArtifacts((prev) => {
          const next = { ...prev };
          for (const a of res.artifacts) next[a.id] = a;
          return next;
        });
        setMessageArtifacts((prev) => ({
          ...prev,
          [res.message.id]: res.artifacts.map((a) => a.id),
        }));
        setOpenArtifactId(res.artifacts[res.artifacts.length - 1].id);
      }
      refreshSessions();
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e);
        // Roll the optimistic turn back — the server never stored it if the
        // request failed before persistence.
        setMessages((m) => m.filter((msg) => msg.id !== optimistic.id));
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleOpenArtifact(id: string) {
    const existing = artifacts[id];
    if (existing && existing.content) {
      setOpenArtifactId(id);
      return;
    }
    try {
      const full = await api.getArtifact(id);
      setArtifacts((prev) => ({ ...prev, [id]: full }));
      setOpenArtifactId(id);
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    }
  }

  async function handleNewChat() {
    try {
      const created = await api.createSession();
      setSessions((s) => [created, ...s]);
      setActiveId(created.id);
      setMessages([]);
      setSidebarOpen(false);
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    }
  }

  async function handleDelete(id: string) {
    try {
      await api.deleteSession(id);
      const remaining = sessions.filter((s) => s.id !== id);
      setSessions(remaining);
      if (activeId === id) {
        if (remaining.length > 0) setActiveId(remaining[0].id);
        else await handleNewChat();
      }
    } catch (e) {
      if (e instanceof ApiError) setError(e);
    }
  }

  const artifactTitles = Object.fromEntries(
    Object.values(artifacts).map((a) => [a.id, a.title]),
  );
  const kbBuilding = config !== null && !config.knowledge_base_ready;

  return (
    <div className="flex h-full">
      <SessionSidebar
        sessions={sessions}
        activeId={activeId}
        config={config}
        open={sidebarOpen}
        onSelect={(id) => {
          setActiveId(id);
          setSidebarOpen(false);
        }}
        onNew={handleNewChat}
        onDelete={handleDelete}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-gray-200 bg-surface px-4 py-2.5 lg:hidden">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open conversation list"
            className="rounded p-1.5 text-ink-soft hover:bg-surface-raised"
          >
            &#9776;
          </button>
          <span className="truncate text-sm font-medium text-ink">
            The Lenny Growth Assistant
          </span>
        </header>

        {kbBuilding && (
          <div
            role="status"
            className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-900"
          >
            Building the knowledge base from Lenny&apos;s Podcast transcripts. Answers
            will be ungrounded until this finishes — this takes a few minutes on first
            start.
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="flex items-start gap-3 border-b border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-900"
          >
            <div className="min-w-0 flex-1">
              <p className="font-semibold">{error.message}</p>
              {error.hint && <p className="mt-0.5">{error.hint}</p>}
              <p className="mt-0.5 font-mono text-[10px] text-red-700">
                {error.code} · request {error.requestId}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
              className="rounded p-1 hover:bg-red-100"
            >
              &#10005;
            </button>
          </div>
        )}

        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-6">
              {booting && (
                <p className="text-center text-sm text-ink-muted">Loading...</p>
              )}

              {!booting && messages.length === 0 && (
                <div className="mx-auto max-w-md py-12 text-center">
                  <h2 className="text-lg font-semibold text-ink">
                    Ask anything about product and growth
                  </h2>
                  <p className="mt-2 text-sm text-ink-muted">
                    Every answer is grounded in transcripts from Lenny&apos;s Podcast,
                    with sources you can check. Ask a question, request a Ship 30 essay,
                    or ask for a document to render beside the chat.
                  </p>
                </div>
              )}

              {messages.map((m) => (
                <MessageBubble
                  key={m.id}
                  message={m}
                  artifactIds={messageArtifacts[m.id]}
                  artifactTitles={artifactTitles}
                  onOpenArtifact={handleOpenArtifact}
                />
              ))}

              {busy && (
                <div className="flex justify-start" role="status" aria-live="polite">
                  <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm bg-surface px-4 py-3 shadow-sm ring-1 ring-gray-100">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="dot h-1.5 w-1.5 rounded-full bg-ink-muted"
                        style={{ animationDelay: `${i * 0.15}s` }}
                      />
                    ))}
                    <span className="sr-only">The assistant is thinking</span>
                  </div>
                </div>
              )}
            </div>

            <div id="composer">
              <Composer disabled={booting || !activeId} busy={busy} onSend={handleSend} />
            </div>
          </div>

          {/* The artifact panel is a right-hand split on desktop and a full
              overlay on narrow screens, where a side-by-side would leave both
              panes unusable. */}
          {openArtifactId && (
            <div className="fixed inset-0 z-40 bg-surface md:static md:z-auto md:w-[45%] md:max-w-2xl">
              <ArtifactViewer
                artifact={artifacts[openArtifactId] || null}
                onClose={() => setOpenArtifactId(null)}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
