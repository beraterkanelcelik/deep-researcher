import { useCallback, useEffect, useRef, useState } from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import {
  convertLangChainMessages,
  type LangChainMessage,
} from "@assistant-ui/react-langgraph";
import { Thread } from "@/components/assistant-ui/thread";
import {
  CalculatorToolUI,
  TimeToolUI,
  SearchToolUI,
  PlanToolUI,
} from "@/components/tools/ToolUIs";
import {
  createThread,
  getThreadState,
  resumeRun,
  sendMessage,
} from "@/lib/chatApi";
import { useAppStore, useNodeStatusStore } from "@/lib/store";
import type { HITLPayload } from "@/lib/store";

interface MyAssistantProps {
  threadId?: string | null;
}

export function MyAssistant({ threadId }: MyAssistantProps) {
  const [messages, setMessages] = useState<LangChainMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const threadIdRef = useRef<string | null>(threadId || null);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const handleNodeEvent = useNodeStatusStore((s) => s.handleNodeEvent);
  const resetNodeStatus = useNodeStatusStore((s) => s.reset);
  const setInterrupt = useNodeStatusStore((s) => s.setInterrupt);

  // Load existing thread messages on mount
  useEffect(() => {
    if (threadId) {
      threadIdRef.current = threadId;
      getThreadState(threadId)
        .then((state) => {
          setMessages(state.values.messages || []);
        })
        .catch(() => {
          setMessages([]);
        });
    }
  }, []); // Only on mount (component re-mounts via key)

  const onInterrupt = useCallback(
    (payload: HITLPayload) => {
      setInterrupt(payload);
    },
    [setInterrupt]
  );

  // Process SSE stream events and update messages state
  const processStream = useCallback(
    async (
      stream: AsyncGenerator<{
        event: string;
        data: LangChainMessage[] | Record<string, unknown>;
      }>
    ) => {
      for await (const event of stream) {
        if (
          event.event === "messages/partial" ||
          event.event === "messages/complete"
        ) {
          const data = event.data as LangChainMessage[];
          if (data.length > 0) {
            setMessages((prev) => {
              let next = [...prev];
              for (const incoming of data) {
                const id = (incoming as Record<string, unknown>).id as
                  | string
                  | undefined;
                if (id) {
                  const idx = next.findIndex(
                    (m) => (m as Record<string, unknown>).id === id
                  );
                  if (idx >= 0) {
                    next = [
                      ...next.slice(0, idx),
                      incoming,
                      ...next.slice(idx + 1),
                    ];
                    continue;
                  }
                }
                next = [...next, incoming];
              }
              return next;
            });
          }
        }
      }
    },
    []
  );

  const handleResume = useCallback(
    async (value: unknown) => {
      if (!threadIdRef.current) return;
      setInterrupt(null);
      setIsRunning(true);

      const stream = resumeRun({
        threadId: threadIdRef.current,
        resumeValue: value,
        onNodeStatus: handleNodeEvent,
        onInterrupt,
      });

      await processStream(stream);
      setIsRunning(false);
    },
    [handleNodeEvent, onInterrupt, setInterrupt, processStream]
  );

  const runtime = useExternalStoreRuntime({
    messages,
    convertMessage: convertLangChainMessages,
    isRunning,
    onNew: async (message) => {
      // Create thread if needed
      if (!threadIdRef.current) {
        const { thread_id } = await createThread();
        threadIdRef.current = thread_id;
      }

      // Convert to LangChain human message
      const text = message.content
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join("\n");

      const humanMsg: LangChainMessage = {
        type: "human",
        content: text,
      };

      setMessages((prev) => [...prev, humanMsg]);
      setIsRunning(true);
      resetNodeStatus();

      const stream = sendMessage({
        threadId: threadIdRef.current!,
        messages: [humanMsg],
        model: selectedModel,
        onNodeStatus: handleNodeEvent,
        onInterrupt,
      });

      await processStream(stream);
      setIsRunning(false);
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onResume={handleResume} />
      <CalculatorToolUI />
      <TimeToolUI />
      <SearchToolUI />
      <PlanToolUI />
    </AssistantRuntimeProvider>
  );
}
