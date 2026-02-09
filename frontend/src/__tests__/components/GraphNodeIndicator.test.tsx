import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { GraphNodeIndicator } from "@/components/GraphNodeIndicator";
import { useNodeStatusStore } from "@/lib/store";

beforeEach(() => {
  useNodeStatusStore.getState().reset();
});

describe("GraphNodeIndicator", () => {
  it("renders nothing when no nodes exist", () => {
    const { container } = render(<GraphNodeIndicator />);
    expect(container.firstChild).toBeNull();
  });

  it("renders core nodes when active", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("retrieve", "active");

    render(<GraphNodeIndicator />);
    expect(screen.getByText("Retrieve")).toBeInTheDocument();
    expect(screen.getByText("Pipeline")).toBeInTheDocument();
  });

  it("renders multiple nodes in sequence", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("retrieve", "completed");
    store.handleNodeEvent("agent", "active");
    store.handleNodeEvent("tools", "active");

    render(<GraphNodeIndicator />);
    expect(screen.getByText("Retrieve")).toBeInTheDocument();
    expect(screen.getByText("Agent")).toBeInTheDocument();
    expect(screen.getByText("Tools")).toBeInTheDocument();
  });

  it("shows completed node with checkmark styling", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("retrieve", "completed");

    render(<GraphNodeIndicator />);
    const retrieveNode = screen.getByText("Retrieve");
    // Completed nodes should have emerald styling
    expect(retrieveNode.closest("div[class*='flex items-center']")).toHaveClass(
      "text-emerald-400"
    );
  });

  it("shows active node with violet styling", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("agent", "active");

    render(<GraphNodeIndicator />);
    const agentNode = screen.getByText("Agent");
    expect(agentNode.closest("div[class*='flex items-center']")).toHaveClass(
      "text-violet-400"
    );
  });

  it("does not show subgraph section when not running", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("retrieve", "active");

    render(<GraphNodeIndicator />);
    expect(screen.queryByText("Subgraph")).not.toBeInTheDocument();
  });

  it("shows subgraph section when subgraph is running", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("retrieve", "completed");
    store.handleNodeEvent("clarify", "active", true);

    render(<GraphNodeIndicator />);
    expect(screen.getByText("Subgraph")).toBeInTheDocument();
    expect(screen.getByText("Clarify")).toBeInTheDocument();
  });

  it("shows multiple subgraph nodes", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("clarify", "completed", true);
    store.handleNodeEvent("orchestrate", "active", true);

    render(<GraphNodeIndicator />);
    expect(screen.getByText("Clarify")).toBeInTheDocument();
    expect(screen.getByText("Plan")).toBeInTheDocument();
  });

  it("shows run count when node runs multiple times", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("agent", "active");
    store.handleNodeEvent("agent", "completed");
    store.handleNodeEvent("agent", "active");
    store.handleNodeEvent("agent", "completed");

    render(<GraphNodeIndicator />);
    expect(screen.getByText("x2")).toBeInTheDocument();
  });

  it("has full opacity when graph is running", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("agent", "active");

    const { container } = render(<GraphNodeIndicator />);
    const wrapper = container.querySelector("div.fixed div.flex.flex-col");
    expect(wrapper).toHaveClass("opacity-100");
  });

  it("has reduced opacity when graph is not running", () => {
    const store = useNodeStatusStore.getState();
    store.handleNodeEvent("__graph__", "started");
    store.handleNodeEvent("agent", "active");
    store.handleNodeEvent("__graph__", "finished");

    const { container } = render(<GraphNodeIndicator />);
    const wrapper = container.querySelector("div.fixed div.flex.flex-col");
    expect(wrapper).toHaveClass("opacity-40");
  });
});
