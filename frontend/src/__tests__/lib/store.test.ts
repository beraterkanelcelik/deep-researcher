import { describe, it, expect, beforeEach } from "vitest";
import { useNodeStatusStore, useAppStore } from "@/lib/store";
import type { HITLPayload } from "@/lib/store";

describe("useAppStore", () => {
  beforeEach(() => {
    const store = useAppStore.getState();
    store.setSidebarOpen(true);
    store.setCurrentThreadId(null);
    store.setSelectedModel("gpt-4.1-mini");
    store.setThreads([]);
  });

  it("toggles sidebar", () => {
    const store = useAppStore.getState();
    expect(store.sidebarOpen).toBe(true);
    store.toggleSidebar();
    expect(useAppStore.getState().sidebarOpen).toBe(false);
    store.toggleSidebar();
    expect(useAppStore.getState().sidebarOpen).toBe(true);
  });

  it("sets current thread id", () => {
    const store = useAppStore.getState();
    store.setCurrentThreadId("thread-123");
    expect(useAppStore.getState().currentThreadId).toBe("thread-123");
  });

  it("sets current thread id to null", () => {
    const store = useAppStore.getState();
    store.setCurrentThreadId("thread-123");
    store.setCurrentThreadId(null);
    expect(useAppStore.getState().currentThreadId).toBeNull();
  });

  it("sets selected model", () => {
    const store = useAppStore.getState();
    store.setSelectedModel("gpt-5-nano-high");
    expect(useAppStore.getState().selectedModel).toBe("gpt-5-nano-high");
  });

  it("sets threads", () => {
    const store = useAppStore.getState();
    const threads = [
      { thread_id: "t1", title: "Thread 1", updated_at: "2025-01-01" },
      { thread_id: "t2", title: "Thread 2", updated_at: "2025-01-02" },
    ];
    store.setThreads(threads);
    expect(useAppStore.getState().threads).toHaveLength(2);
    expect(useAppStore.getState().threads[0].thread_id).toBe("t1");
  });
});

describe("useNodeStatusStore", () => {
  beforeEach(() => {
    useNodeStatusStore.getState().reset();
  });

  it("handles __graph__ start event", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    const state = useNodeStatusStore.getState();
    expect(state.isGraphRunning).toBe(true);
    expect(state.nodes).toHaveLength(0);
    expect(state.subgraphNodes).toHaveLength(0);
  });

  it("handles __graph__ finished event", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("__graph__", "finished");
    const state = useNodeStatusStore.getState();
    expect(state.isGraphRunning).toBe(false);
    expect(state.isSubgraphRunning).toBe(false);
  });

  it("adds main graph node on active event", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("retrieve", "active");
    const state = useNodeStatusStore.getState();
    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0].name).toBe("retrieve");
    expect(state.nodes[0].status).toBe("active");
    expect(state.nodes[0].runCount).toBe(1);
  });

  it("completes main graph node", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("retrieve", "active");
    store.handleNodeEvent("retrieve", "completed");
    const state = useNodeStatusStore.getState();
    expect(state.nodes[0].status).toBe("completed");
  });

  it("handles subgraph node events", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("clarify", "active", true);
    const state = useNodeStatusStore.getState();
    expect(state.subgraphNodes).toHaveLength(1);
    expect(state.subgraphNodes[0].name).toBe("clarify");
    expect(state.subgraphNodes[0].status).toBe("active");
    expect(state.isSubgraphRunning).toBe(true);
  });

  it("sets isSubgraphRunning when subgraph node becomes active", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("clarify", "active", true);
    expect(useNodeStatusStore.getState().isSubgraphRunning).toBe(true);
  });

  it("increments runCount on repeated activations", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("agent", "active");
    store.handleNodeEvent("agent", "completed");
    store.handleNodeEvent("agent", "active");
    const state = useNodeStatusStore.getState();
    expect(state.nodes[0].runCount).toBe(2);
  });

  it("sets activeInterrupt", () => {
    const payload: HITLPayload = {
      hitl_type: "checkbox",
      title: "Test",
      message: "Test message",
      options: [{ id: "1", label: "Option 1", description: "", selected: false }],
      report: null,
    };
    const store = useNodeStatusStore.getState();
    store.setInterrupt(payload);
    expect(useNodeStatusStore.getState().activeInterrupt).toEqual(payload);
  });

  it("clears activeInterrupt", () => {
    const store = useNodeStatusStore.getState();
    store.setInterrupt({
      hitl_type: "yes_no",
      title: "Test",
      message: "msg",
      options: [],
      report: null,
    });
    store.setInterrupt(null);
    expect(useNodeStatusStore.getState().activeInterrupt).toBeNull();
  });

  it("resets all state", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("retrieve", "active");
    store.handleNodeEvent("clarify", "active", true);
    store.setInterrupt({
      hitl_type: "text",
      title: "T",
      message: "M",
      options: [],
      report: null,
    });
    store.reset();
    const state = useNodeStatusStore.getState();
    expect(state.nodes).toHaveLength(0);
    expect(state.subgraphNodes).toHaveLength(0);
    expect(state.isGraphRunning).toBe(false);
    expect(state.isSubgraphRunning).toBe(false);
    expect(state.activeInterrupt).toBeNull();
  });

  it("__graph__ start resets previous nodes", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("retrieve", "active");
    store.handleNodeEvent("agent", "active");
    store.handleNodeEvent("__graph__", "started");
    const state = useNodeStatusStore.getState();
    expect(state.nodes).toHaveLength(0);
    expect(state.subgraphNodes).toHaveLength(0);
  });
});
