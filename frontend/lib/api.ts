import type {
  AppConfig,
  ApiErrorBody,
  Artifact,
  ChatResponse,
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

  getArtifact: (id: string) => request<Artifact>(`/api/artifacts/${id}`),
};
