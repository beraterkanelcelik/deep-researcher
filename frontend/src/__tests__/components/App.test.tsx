import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

// Mock external modules before importing App
vi.mock("@/lib/chatApi", () => ({
  checkHealth: vi.fn(),
  createThread: vi.fn(),
  listThreads: vi.fn().mockResolvedValue([]),
  getThreadState: vi.fn(),
  sendMessage: vi.fn(),
  resumeRun: vi.fn(),
  listDocuments: vi.fn().mockResolvedValue([]),
  uploadDocument: vi.fn(),
  deleteDocument: vi.fn(),
  deleteThread: vi.fn(),
}));

vi.mock("@/components/MyAssistant", () => ({
  MyAssistant: ({ threadId }: { threadId?: string | null }) => (
    <div data-testid="my-assistant">MyAssistant {threadId || "new"}</div>
  ),
}));

import App from "@/App";
import { checkHealth } from "@/lib/chatApi";

const mockCheckHealth = vi.mocked(checkHealth);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("App", () => {
  it("shows loading state while waiting for backend", () => {
    mockCheckHealth.mockResolvedValue(false);
    render(<App />);
    expect(
      screen.getByText("Waiting for services to start...")
    ).toBeInTheDocument();
  });

  it("shows main UI after health check passes", async () => {
    mockCheckHealth.mockResolvedValue(true);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("AI Chat")).toBeInTheDocument();
    });
  });

  it("renders model selector after ready", async () => {
    mockCheckHealth.mockResolvedValue(true);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByDisplayValue("GPT-4.1 Mini")).toBeInTheDocument();
    });
  });

  it("renders MyAssistant component after ready", async () => {
    mockCheckHealth.mockResolvedValue(true);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("my-assistant")).toBeInTheDocument();
    });
  });
});
