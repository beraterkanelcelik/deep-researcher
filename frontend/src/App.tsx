import { useState, useEffect } from "react";
import { MyAssistant } from "@/components/MyAssistant";
import { ThreadList } from "@/components/ThreadList";
import { DocumentUpload } from "@/components/DocumentUpload";
import { PanelLeftClose, PanelLeft, Loader2 } from "lucide-react";
import { checkHealth } from "@/lib/chatApi";
import { useAppStore } from "@/lib/store";

const MODEL_OPTIONS = [
  { label: "GPT-4.1 Mini", value: "gpt-4.1-mini" },
  { label: "GPT-5 Nano High", value: "gpt-5-nano-high" },
  { label: "GPT-5 Nano Medium", value: "gpt-5-nano-medium" },
  { label: "GPT-5 Nano Low", value: "gpt-5-nano-low" },
  { label: "GPT-5 Nano Minimal", value: "gpt-5-nano-minimal" },
];

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [backendReady, setBackendReady] = useState(false);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      while (!cancelled) {
        const ok = await checkHealth();
        if (ok && !cancelled) {
          setBackendReady(true);
          return;
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };

    poll();
    return () => { cancelled = true; };
  }, []);

  // Force re-mount MyAssistant when thread changes by using key
  const assistantKey = currentThreadId || "new";

  if (!backendReady) {
    return (
      <div className="flex items-center justify-center h-screen bg-zinc-950">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-zinc-400 mx-auto mb-4" />
          <p className="text-zinc-400 text-sm">Waiting for services to start...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } flex-shrink-0 border-r border-zinc-800 bg-zinc-950 transition-all duration-200 overflow-hidden`}
      >
        <div className="flex flex-col h-full w-64">
          <ThreadList
            currentThreadId={currentThreadId}
            onSelectThread={(id) => setCurrentThreadId(id)}
            onNewThread={(id) => setCurrentThreadId(id)}
          />
          <DocumentUpload />
        </div>
      </div>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center h-12 px-3 border-b border-zinc-800 flex-shrink-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            {sidebarOpen ? (
              <PanelLeftClose className="h-4 w-4" />
            ) : (
              <PanelLeft className="h-4 w-4" />
            )}
          </button>
          <span className="ml-3 text-sm text-zinc-500">AI Chat</span>
          <div className="ml-auto">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-zinc-900 text-zinc-300 text-sm border border-zinc-700 rounded-md px-2 py-1 focus:outline-none focus:border-zinc-500"
            >
              {MODEL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Chat */}
        <div className="flex-1 overflow-hidden">
          <MyAssistant key={assistantKey} threadId={currentThreadId} />
        </div>
      </div>
    </div>
  );
}
