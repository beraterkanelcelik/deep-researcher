import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Import after mocking
import {
  checkHealth,
  createThread,
  listThreads,
  deleteThread,
} from "@/lib/chatApi";

beforeEach(() => {
  mockFetch.mockReset();
});

describe("checkHealth", () => {
  it("returns true when backend is healthy", async () => {
    mockFetch.mockResolvedValue({ ok: true });
    const result = await checkHealth();
    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining("/api/health"));
  });

  it("returns false when backend is down", async () => {
    mockFetch.mockRejectedValue(new Error("Network error"));
    const result = await checkHealth();
    expect(result).toBe(false);
  });

  it("returns false when response is not ok", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 500 });
    const result = await checkHealth();
    expect(result).toBe(false);
  });
});

describe("createThread", () => {
  it("returns thread id on success", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ thread_id: "abc-123" }),
    });
    const result = await createThread();
    expect(result.thread_id).toBe("abc-123");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/threads/"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("throws on failure", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 500 });
    await expect(createThread()).rejects.toThrow("Failed to create thread");
  });
});

describe("listThreads", () => {
  it("returns array of threads", async () => {
    const threads = [
      { thread_id: "t1", title: "Thread 1", updated_at: "2025-01-01" },
      { thread_id: "t2", title: "Thread 2", updated_at: "2025-01-02" },
    ];
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(threads),
    });
    const result = await listThreads();
    expect(result).toHaveLength(2);
    expect(result[0].thread_id).toBe("t1");
  });
});

describe("deleteThread", () => {
  it("calls DELETE endpoint", async () => {
    mockFetch.mockResolvedValue({ ok: true });
    await deleteThread("thread-123");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/threads/thread-123/"),
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("throws on failure", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 404 });
    await expect(deleteThread("bad-id")).rejects.toThrow("Failed to delete thread");
  });
});

describe("parseSSEStream (via sendMessage)", () => {
  function makeSSEResponse(events: Array<{ event: string; data: string }>) {
    const text = events
      .map((e) => `event: ${e.event}\ndata: ${e.data}\n\n`)
      .join("");
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(text));
        controller.close();
      },
    });
    return {
      ok: true,
      body: stream,
      headers: new Headers({ "content-type": "text/event-stream" }),
    };
  }

  it("yields metadata events", async () => {
    const { sendMessage } = await import("@/lib/chatApi");
    mockFetch.mockResolvedValue(
      makeSSEResponse([
        { event: "metadata", data: JSON.stringify({ run_id: "run-1" }) },
        { event: "end", data: "{}" },
      ])
    );

    const events = [];
    for await (const event of sendMessage({
      threadId: "t1",
      messages: [],
    })) {
      events.push(event);
    }
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe("metadata");
  });

  it("yields messages/partial events", async () => {
    const { sendMessage } = await import("@/lib/chatApi");
    const msgData = [{ type: "ai", content: "Hello", id: "msg-1" }];
    mockFetch.mockResolvedValue(
      makeSSEResponse([
        { event: "messages/partial", data: JSON.stringify(msgData) },
        { event: "end", data: "{}" },
      ])
    );

    const events = [];
    for await (const event of sendMessage({
      threadId: "t1",
      messages: [],
    })) {
      events.push(event);
    }
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe("messages/partial");
  });

  it("dispatches node/status via callback, not yielded", async () => {
    const { sendMessage } = await import("@/lib/chatApi");
    const onNodeStatus = vi.fn();

    mockFetch.mockResolvedValue(
      makeSSEResponse([
        {
          event: "node/status",
          data: JSON.stringify({ node: "agent", status: "active" }),
        },
        { event: "end", data: "{}" },
      ])
    );

    const events = [];
    for await (const event of sendMessage({
      threadId: "t1",
      messages: [],
      onNodeStatus,
    })) {
      events.push(event);
    }
    // node/status should NOT be yielded
    expect(events).toHaveLength(0);
    // But callback should have been called
    expect(onNodeStatus).toHaveBeenCalledWith("agent", "active", false);
  });

  it("dispatches interrupt via callback, not yielded", async () => {
    const { sendMessage } = await import("@/lib/chatApi");
    const onInterrupt = vi.fn();

    const interruptPayload = {
      hitl_type: "checkbox",
      title: "Select",
      message: "Pick topics",
      options: [],
    };

    mockFetch.mockResolvedValue(
      makeSSEResponse([
        { event: "interrupt", data: JSON.stringify(interruptPayload) },
        { event: "end", data: "{}" },
      ])
    );

    const events = [];
    for await (const event of sendMessage({
      threadId: "t1",
      messages: [],
      onInterrupt,
    })) {
      events.push(event);
    }
    expect(events).toHaveLength(0);
    expect(onInterrupt).toHaveBeenCalledWith(interruptPayload);
  });

  it("handles subgraph flag in node/status", async () => {
    const { sendMessage } = await import("@/lib/chatApi");
    const onNodeStatus = vi.fn();

    mockFetch.mockResolvedValue(
      makeSSEResponse([
        {
          event: "node/status",
          data: JSON.stringify({
            node: "clarify",
            status: "active",
            subgraph: true,
          }),
        },
        { event: "end", data: "{}" },
      ])
    );

    for await (const _ of sendMessage({
      threadId: "t1",
      messages: [],
      onNodeStatus,
    })) {
      // consume
    }
    expect(onNodeStatus).toHaveBeenCalledWith("clarify", "active", true);
  });
});
