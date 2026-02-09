import type { LangChainMessage } from "@assistant-ui/react-langgraph";
import type { HITLPayload } from "./store";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export const checkHealth = async (): Promise<boolean> => {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
};

export const createThread = async (): Promise<{ thread_id: string }> => {
  const res = await fetch(`${API_BASE}/api/threads/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(`Failed to create thread: ${res.status}`);
  return res.json();
};

export const getThreadState = async (
  threadId: string
): Promise<{
  values: { messages: LangChainMessage[] };
  tasks: Array<{ interrupts?: unknown[] }>;
}> => {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/state`);
  if (!res.ok) throw new Error(`Failed to get thread state: ${res.status}`);
  return res.json();
};

export const listThreads = async (): Promise<
  Array<{
    thread_id: string;
    title: string;
    updated_at: string;
  }>
> => {
  const res = await fetch(`${API_BASE}/api/threads/`);
  if (!res.ok) throw new Error(`Failed to list threads: ${res.status}`);
  return res.json();
};

export const deleteThread = async (threadId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to delete thread: ${res.status}`);
};

interface LangGraphMessagesEvent {
  event: string;
  data: LangChainMessage[] | Record<string, unknown>;
}

interface SSECallbacks {
  onNodeStatus?: (node: string, status: string, isSubgraph?: boolean) => void;
  onInterrupt?: (payload: HITLPayload) => void;
}

/**
 * Shared SSE stream parser. Reads from a Response and yields LangGraph-compatible events.
 * Side-channel events (node/status, interrupt) are dispatched via callbacks, not yielded.
 */
async function* parseSSEStream(
  res: Response,
  callbacks: SSECallbacks
): AsyncGenerator<LangGraphMessagesEvent> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      let currentEvent = "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          const dataStr = line.slice(6).trim();

          if (currentEvent === "end") {
            return;
          }

          if (currentEvent === "error") {
            try {
              const errorData = JSON.parse(dataStr);
              throw new Error(errorData.content || "Stream error");
            } catch (e) {
              if (e instanceof SyntaxError) throw new Error("Stream error");
              throw e;
            }
          }

          if (currentEvent === "node/status") {
            try {
              const nodeData = JSON.parse(dataStr);
              callbacks.onNodeStatus?.(
                nodeData.node,
                nodeData.status,
                nodeData.subgraph === true
              );
            } catch {
              // Skip malformed JSON
            }
            currentEvent = "";
            continue;
          }

          if (currentEvent === "interrupt") {
            try {
              const interruptData = JSON.parse(dataStr) as HITLPayload;
              callbacks.onInterrupt?.(interruptData);
            } catch {
              // Skip malformed JSON
            }
            currentEvent = "";
            continue;
          }

          if (
            currentEvent === "messages/partial" ||
            currentEvent === "messages/complete" ||
            currentEvent === "metadata"
          ) {
            try {
              const data = JSON.parse(dataStr);
              yield { event: currentEvent, data };
            } catch {
              // Skip malformed JSON
            }
          }

          currentEvent = "";
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Send a message and stream the response via SSE.
 */
export async function* sendMessage(params: {
  threadId: string;
  messages: LangChainMessage[];
  model?: string;
  abortSignal?: AbortSignal;
  onNodeStatus?: (node: string, status: string, isSubgraph?: boolean) => void;
  onInterrupt?: (payload: HITLPayload) => void;
}): AsyncGenerator<LangGraphMessagesEvent> {
  const res = await fetch(
    `${API_BASE}/api/threads/${params.threadId}/runs/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: { messages: params.messages },
        model: params.model,
        stream_mode: "messages",
        config: {},
      }),
      signal: params.abortSignal,
    }
  );

  if (!res.ok) {
    throw new Error(`Stream request failed: ${res.status}`);
  }

  yield* parseSSEStream(res, {
    onNodeStatus: params.onNodeStatus,
    onInterrupt: params.onInterrupt,
  });
}

/**
 * Resume a graph run after an HITL interrupt.
 * Sends the user's response value and streams the continued execution.
 */
export async function* resumeRun(params: {
  threadId: string;
  resumeValue: unknown;
  abortSignal?: AbortSignal;
  onNodeStatus?: (node: string, status: string, isSubgraph?: boolean) => void;
  onInterrupt?: (payload: HITLPayload) => void;
}): AsyncGenerator<LangGraphMessagesEvent> {
  const res = await fetch(
    `${API_BASE}/api/threads/${params.threadId}/runs/resume`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resume_value: params.resumeValue,
      }),
      signal: params.abortSignal,
    }
  );

  if (!res.ok) {
    throw new Error(`Resume request failed: ${res.status}`);
  }

  yield* parseSSEStream(res, {
    onNodeStatus: params.onNodeStatus,
    onInterrupt: params.onInterrupt,
  });
}

// Document API
export const uploadDocument = async (
  file: File
): Promise<{ document_id: string; chunks: number; status: string }> => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Failed to upload document: ${res.status}`);
  return res.json();
};

export const listDocuments = async (): Promise<
  Array<{
    id: string;
    filename: string;
    chunks: number;
    created_at: string;
  }>
> => {
  const res = await fetch(`${API_BASE}/api/documents/`);
  if (!res.ok) throw new Error(`Failed to list documents: ${res.status}`);
  return res.json();
};

export const deleteDocument = async (documentId: string): Promise<void> => {
  const res = await fetch(`${API_BASE}/api/documents/${documentId}/`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to delete document: ${res.status}`);
};
