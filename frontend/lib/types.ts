export type ArtifactKind = "markdown" | "html";

export interface Citation {
  chunk_id: string;
  episode_title: string;
  guest: string;
  timestamp: string;
  url: string;
  publish_date: string | null;
  score: number;
  similarity: number;
}

export interface Artifact {
  id: string;
  kind: ArtifactKind;
  title: string;
  content: string;
  created_at: string;
  sanitizer_report?: {
    sanitizer: string;
    modified: boolean;
    findings: string[];
    original_bytes: number;
    sanitized_bytes: number;
  } | null;
}

export interface ArtifactSummary {
  id: string;
  kind: ArtifactKind;
  title: string;
  created_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  provider: string | null;
  model: string | null;
  citations: Citation[];
  latency_ms: number | null;
  created_at: string;
}

export interface ToolCall {
  tool: string;
  ok: boolean;
  latency_ms: number;
}

export interface ChatResponse {
  session_id: string;
  message: Message;
  artifacts: Artifact[];
  tool_calls: ToolCall[];
  grounded: boolean;
  provider: string;
  model: string;
  latency_ms: number;
}

export interface SessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionDetail extends SessionSummary {
  messages: Message[];
  artifacts: ArtifactSummary[];
}

export interface AppConfig {
  provider: string;
  model: string;
  endpoint: string;
  is_local: boolean;
  api_key_required: boolean;
  api_key_present: boolean;
  fallback_provider: string;
  knowledge_base_ready: boolean;
  episodes: number;
  chunks: number;
}

/** The single error envelope every endpoint returns. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    hint: string;
    request_id: string;
    fields?: { loc: string; msg: string }[];
  };
}
