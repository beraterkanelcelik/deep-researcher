import { useEffect, useState, useCallback } from "react";
import { listThreads, deleteThread, createThread } from "@/lib/chatApi";
import { MessageSquarePlus, Trash2, MessageSquare } from "lucide-react";

interface ThreadItem {
  thread_id: string;
  title: string;
  updated_at: string;
}

interface ThreadListProps {
  currentThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onNewThread: (threadId: string) => void;
}

export function ThreadList({
  currentThreadId,
  onSelectThread,
  onNewThread,
}: ThreadListProps) {
  const [threads, setThreads] = useState<ThreadItem[]>([]);
  const [loading, setLoading] = useState(true);

  const loadThreads = useCallback(async () => {
    try {
      const data = await listThreads();
      setThreads(data);
    } catch (err) {
      console.error("Failed to load threads:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadThreads();
    const interval = setInterval(loadThreads, 5000);
    return () => clearInterval(interval);
  }, [loadThreads]);

  const handleNewThread = async () => {
    try {
      const { thread_id } = await createThread();
      onNewThread(thread_id);
      loadThreads();
    } catch (err) {
      console.error("Failed to create thread:", err);
    }
  };

  const handleDelete = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    try {
      await deleteThread(threadId);
      loadThreads();
    } catch (err) {
      console.error("Failed to delete thread:", err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-zinc-800">
        <button
          onClick={handleNewThread}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-sm transition-colors"
        >
          <MessageSquarePlus className="h-4 w-4" />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {loading && threads.length === 0 ? (
          <div className="text-zinc-600 text-xs text-center py-8">
            Loading...
          </div>
        ) : threads.length === 0 ? (
          <div className="text-zinc-600 text-xs text-center py-8">
            No conversations yet
          </div>
        ) : (
          <div className="space-y-1">
            {threads.map((thread) => (
              <div
                key={thread.thread_id}
                onClick={() => onSelectThread(thread.thread_id)}
                className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                  currentThreadId === thread.thread_id
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
                }`}
              >
                <MessageSquare className="h-4 w-4 flex-shrink-0" />
                <span className="truncate flex-1">
                  {thread.title || "New conversation"}
                </span>
                <button
                  onClick={(e) => handleDelete(e, thread.thread_id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-all"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
