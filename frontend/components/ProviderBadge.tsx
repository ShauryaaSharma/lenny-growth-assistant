"use client";

import type { AppConfig } from "@/lib/types";

/**
 * Shows which model actually served the conversation. Required by the brief
 * (3.2: "make the selected provider visible in the UI"), but it also matters
 * operationally: when fallback fires, the badge is how you notice.
 */
export default function ProviderBadge({ config }: { config: AppConfig | null }) {
  if (!config) {
    return (
      <span className="rounded-full bg-surface-raised px-2.5 py-1 text-xs text-ink-muted">
        Connecting...
      </span>
    );
  }

  const healthy = config.is_local || config.api_key_present;
  const label = config.is_local ? "Local" : "Cloud";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
        healthy ? "bg-accent-soft text-accent" : "bg-amber-50 text-amber-800"
      }`}
      title={`${config.provider} at ${config.endpoint}${
        healthy ? "" : " — no API key configured"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${healthy ? "bg-accent" : "bg-amber-500"}`}
        aria-hidden="true"
      />
      {label} · {config.model}
    </span>
  );
}
