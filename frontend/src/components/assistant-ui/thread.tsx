import {
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { SendHorizontal } from "lucide-react";
import { GraphNodeIndicator } from "@/components/GraphNodeIndicator";
import { HITLWidget } from "@/components/HITLWidgets";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { useNodeStatusStore } from "@/lib/store";

interface ThreadProps {
  onResume?: (value: unknown) => void;
}

export function Thread({ onResume }: ThreadProps) {
  const activeInterrupt = useNodeStatusStore((s) => s.activeInterrupt);

  return (
    <>
      <GraphNodeIndicator />
      <ThreadPrimitive.Root className="flex flex-col h-full">
        <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
          <ThreadPrimitive.Empty>
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md px-4">
                <h2 className="text-2xl font-semibold text-zinc-200 mb-2">
                  AI Research Assistant
                </h2>
                <p className="text-zinc-500 text-sm">
                  Ask me anything. I can help with questions, calculations,
                  search through your documents, and conduct deep research on
                  any topic with parallel web searches and structured reports.
                </p>
              </div>
            </div>
          </ThreadPrimitive.Empty>

          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />

          {/* Render HITL widget when an interrupt is active */}
          {activeInterrupt && onResume && (
            <HITLWidget payload={activeInterrupt} onResume={onResume} />
          )}
        </ThreadPrimitive.Viewport>

        <Composer disabled={!!activeInterrupt} />
      </ThreadPrimitive.Root>
    </>
  );
}

function Composer({ disabled }: { disabled?: boolean }) {
  return (
    <ComposerPrimitive.Root className="mx-auto w-full max-w-2xl px-4 pb-4">
      <div
        className={`flex items-end gap-2 rounded-2xl border border-zinc-800 bg-zinc-900 p-2 ${
          disabled ? "opacity-50 pointer-events-none" : ""
        }`}
      >
        <ComposerPrimitive.Input
          placeholder={
            disabled
              ? "Respond to the prompt above first..."
              : "Type a message..."
          }
          className="flex-1 bg-transparent resize-none outline-none text-sm text-zinc-100 placeholder:text-zinc-600 min-h-[36px] max-h-[200px] px-2 py-1.5"
          autoFocus
          disabled={disabled}
        />
        <ComposerPrimitive.Send
          className="flex items-center justify-center h-8 w-8 rounded-lg bg-white text-zinc-900 hover:bg-zinc-200 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          disabled={disabled}
        >
          <SendHorizontal className="h-4 w-4" />
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end px-4 py-2 max-w-2xl mx-auto w-full">
      <div className="bg-zinc-800 text-zinc-100 rounded-2xl rounded-br-md px-4 py-2.5 max-w-[80%] text-sm leading-relaxed">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-start px-4 py-2 max-w-2xl mx-auto w-full">
      <div className="flex gap-3 max-w-[85%]">
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-violet-600 to-blue-600 flex items-center justify-center text-xs font-bold text-white mt-0.5">
          AI
        </div>
        <div className="text-zinc-300 text-sm leading-relaxed pt-1 min-w-0 flex-1">
          <MessagePrimitive.Content
            components={{
              tools: {
                Fallback: ToolFallback,
              },
            }}
          />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}
