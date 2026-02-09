import type { ToolCallContentPartComponent } from "@assistant-ui/react";
import { useState } from "react";
import {
  ChevronRight,
  ChevronDown,
  Check,
  Loader2,
  Calculator,
  Clock,
  Search,
  ListTodo,
  Microscope,
  Save,
  Wrench,
  AlertCircle,
} from "lucide-react";

const toolConfig: Record<
  string,
  { label: string; icon: React.ComponentType<{ className?: string }> }
> = {
  get_current_time: { label: "Get Current Time", icon: Clock },
  calculator: { label: "Calculator", icon: Calculator },
  tavily_search: { label: "Web Search", icon: Search },
  create_plan: { label: "Create Plan", icon: ListTodo },
  deep_research: { label: "Deep Research", icon: Microscope },
  save_report: { label: "Save Report", icon: Save },
};

function getToolConfig(name: string) {
  if (toolConfig[name]) return toolConfig[name];
  const label = name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return { label, icon: Wrench };
}

export const ToolFallback: ToolCallContentPartComponent = ({
  toolName,
  argsText,
  result,
  status,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const config = getToolConfig(toolName);
  const Icon = config.icon;
  const isRunning = status.type === "running";
  const isComplete = status.type === "complete";
  const isError =
    status.type === "incomplete" && status.reason === "error";

  return (
    <div className="my-1.5 rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden text-sm">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2.5 w-full px-3 py-2 text-left hover:bg-zinc-800/40 transition-colors"
      >
        {/* Status icon */}
        <div className="flex-shrink-0">
          {isRunning ? (
            <Loader2 className="h-3.5 w-3.5 text-violet-400 animate-spin" />
          ) : isError ? (
            <AlertCircle className="h-3.5 w-3.5 text-red-400" />
          ) : isComplete ? (
            <Check className="h-3.5 w-3.5 text-emerald-400" />
          ) : (
            <Icon className="h-3.5 w-3.5 text-zinc-500" />
          )}
        </div>

        {/* Tool icon + name */}
        <div className="flex items-center gap-1.5 min-w-0">
          <Icon className="h-3.5 w-3.5 text-zinc-500 flex-shrink-0" />
          <span
            className={`text-xs font-medium ${
              isRunning
                ? "text-violet-300"
                : isError
                  ? "text-red-300"
                  : isComplete
                    ? "text-zinc-300"
                    : "text-zinc-500"
            }`}
          >
            {config.label}
          </span>
        </div>

        {/* Running shimmer bar */}
        {isRunning && (
          <div className="flex-1 max-w-[80px] h-1 rounded-full bg-zinc-800 overflow-hidden">
            <div className="h-full w-full rounded-full bg-gradient-to-r from-transparent via-violet-500/50 to-transparent animate-pulse" />
          </div>
        )}

        <div className="flex-1" />

        {/* Expand toggle */}
        <div className="flex-shrink-0 text-zinc-600">
          {isExpanded ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </div>
      </button>

      {/* Collapsible detail panel */}
      {isExpanded && (
        <div className="border-t border-zinc-800 px-3 py-2 space-y-2">
          {argsText && argsText !== "{}" && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">
                Arguments
              </p>
              <pre className="text-xs text-zinc-400 whitespace-pre-wrap break-all bg-zinc-950/50 rounded px-2 py-1.5">
                {formatArgs(argsText)}
              </pre>
            </div>
          )}
          {result !== undefined && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">
                Result
              </p>
              <pre className="text-xs text-zinc-400 whitespace-pre-wrap break-all bg-zinc-950/50 rounded px-2 py-1.5 max-h-[200px] overflow-y-auto">
                {formatResult(result)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

function formatArgs(argsText: string): string {
  try {
    return JSON.stringify(JSON.parse(argsText), null, 2);
  } catch {
    return argsText;
  }
}

function formatResult(result: unknown): string {
  if (typeof result === "string") {
    try {
      return JSON.stringify(JSON.parse(result), null, 2);
    } catch {
      return result;
    }
  }
  return JSON.stringify(result, null, 2);
}
