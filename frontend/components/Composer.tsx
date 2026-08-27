"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  disabled: boolean;
  busy: boolean;
  onSend: (message: string) => void;
}

const SUGGESTIONS = [
  "How do you know when you've found product-market fit?",
  "Write a Ship 30 essay about retention as a growth lever",
  "Build me a one-page onboarding audit checklist",
];

export default function Composer({ disabled, busy, onSend }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with content, up to a cap, so long questions stay readable.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  function submit() {
    const text = value.trim();
    if (!text || busy || disabled) return;
    onSend(text);
    setValue("");
  }

  return (
    <div className="border-t border-gray-200 bg-surface px-4 py-3">
      {value === "" && !busy && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setValue(s)}
              disabled={disabled}
              className="rounded-full border border-gray-200 px-3 py-1 text-xs text-ink-muted transition hover:border-accent-ring hover:bg-accent-soft hover:text-accent disabled:opacity-50"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        <label htmlFor="composer" className="sr-only">
          Message
        </label>
        <textarea
          id="composer"
          ref={ref}
          rows={1}
          value={value}
          disabled={disabled}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends; Shift+Enter is a newline. Standard chat convention.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={
            disabled
              ? "Waiting for the assistant to come online..."
              : "Ask about product, growth, retention, pricing, hiring..."
          }
          className="max-h-[200px] flex-1 resize-none rounded-xl border border-gray-200 px-3.5 py-2.5 text-[15px] leading-relaxed text-ink placeholder:text-ink-muted focus:border-accent-ring disabled:bg-surface-sunken"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || busy || value.trim() === ""}
          aria-label="Send message"
          className="mb-0.5 rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          {busy ? "..." : "Send"}
        </button>
      </div>

      <p className="mt-1.5 text-[11px] text-ink-muted">
        Answers are grounded in Lenny&apos;s Podcast transcripts. Enter to send,
        Shift+Enter for a new line.
      </p>
    </div>
  );
}
