import type {
  AppConfig,
  ApiErrorBody,
  Artifact,
  ChatResponse,
  ProgressEvent,
  SessionDetail,
  SessionSummary,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/**
 * Carries the backend's typed error envelope through to the UI, so the user
 * sees "Ollama isn't running, start it with `ollama serve`" rather than a
 * generic failure toast. `hint` is what makes an error actionable.
 */
export class ApiError extends Error {
  code: string;
  hint: string;
  requestId: string;
  status: number;

  constructor(status: number, body?: ApiErrorBody) {
    super(body?.error?.message || "Request failed");
    this.name = "ApiError";
    this.status = status;
    this.code = body?.error?.code || "unknown_error";
    this.hint = body?.error?.hint || "";
    this.requestId = body?.error?.request_id || "-";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
  } catch {
    // Network-level failure: the API itself is unreachable.
    throw new ApiError(0, {
      error: {
        code: "api_unreachable",
        message: `Cannot reach the API at ${BASE}.`,
        hint: "Is the backend running? Try: docker compose up backend",
        request_id: "-",
      },
    });
  }

  if (res.status === 204) return undefined as T;

  if (!res.ok) {
    let body: ApiErrorBody | undefined;
    try {
      body = (await res.json()) as ApiErrorBody;
    } catch {
      body = undefined;
    }
    throw new ApiError(res.status, body);
  }

  return (await res.json()) as T;
}

const UNREACHABLE: ApiErrorBody = {
  error: {
    code: "api_unreachable",
    message: `Cannot reach the API at ${BASE}.`,
    hint: "Is the backend running? Try: docker compose up backend",
    request_id: "-",
  },
};

/**
 * One chat turn over Server-Sent Events. Calls `onProgress` for each step the
 * agent takes and resolves with the same ChatResponse `/chat` returns, once
 * the answer has passed the grounding guards and been saved.
 *
 * fetch rather than EventSource: EventSource can only GET, and the message
 * belongs in a POST body.
 */
async function streamMessage(
  sessionId: string,
  message: string,
  onProgress: (event: ProgressEvent) => void,
): Promise<ChatResponse> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/sessions/${sessionId}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message }),
    });
  } catch {
    throw new ApiError(0, UNREACHABLE);
  }

  // Validation errors and a missing session arrive as ordinary JSON.
  if (!res.ok || !res.body) {
    let body: ApiErrorBody | undefined;
    try {
      body = (await res.json()) as ApiErrorBody;
    } catch {
      body = undefined;
    }
    throw new ApiError(res.status, body);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (!data) continue; // a keepalive comment

      const event = JSON.parse(data);
      if (event.type === "done") return event.response as ChatResponse;
      if (event.type === "error") throw new ApiError(event.status, { error: event.error });
      onProgress(event as ProgressEvent);
    }
  }

  // The connection dropped mid-turn. The server keeps going and saves the
  // answer regardless, so the useful advice is to look again, not to resend.
  throw new ApiError(0, {
    error: {
      code: "stream_interrupted",
      message: "The connection closed before the answer arrived.",
      hint: "The answer is still being saved. Reopen this chat in a moment to see it.",
      request_id: "-",
    },
  });
}

export const api = {
  getConfig: () => request<AppConfig>("/api/config"),

  listSessions: () => request<SessionSummary[]>("/api/sessions"),

  createSession: () =>
    request<SessionSummary>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    }),

  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),

  deleteSession: (id: string) =>
    request<void>(`/api/sessions/${id}`, { method: "DELETE" }),

  sendMessage: (sessionId: string, message: string) =>
    request<ChatResponse>(`/api/sessions/${sessionId}/chat`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  streamMessage,

  getArtifact: (id: string) => request<Artifact>(`/api/artifacts/${id}`),
};
