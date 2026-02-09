import { makeAssistantToolUI } from "@assistant-ui/react";
import {
  Calculator,
  Clock,
  Search,
  ExternalLink,
  Loader2,
  ListTodo,
  CheckCircle2,
  Circle,
} from "lucide-react";

/**
 * Calculator tool — shows expression = result
 */
export const CalculatorToolUI = makeAssistantToolUI<
  { expression: string },
  string
>({
  toolName: "calculator",
  render: ({ args, result, status }) => {
    const isRunning = status.type === "running";

    return (
      <div className="my-1.5 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
        <Calculator className="h-4 w-4 text-violet-400 flex-shrink-0" />
        <code className="text-xs text-zinc-300">{args.expression}</code>
        {isRunning ? (
          <Loader2 className="h-3.5 w-3.5 text-violet-400 animate-spin ml-auto" />
        ) : result !== undefined ? (
          <>
            <span className="text-zinc-600 text-xs">=</span>
            <span className="text-sm font-semibold text-emerald-400">
              {result}
            </span>
          </>
        ) : null}
      </div>
    );
  },
});

/**
 * Get Current Time tool — shows formatted time
 */
export const TimeToolUI = makeAssistantToolUI<Record<string, never>, string>({
  toolName: "get_current_time",
  render: ({ result, status }) => {
    const isRunning = status.type === "running";

    return (
      <div className="my-1.5 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
        <Clock className="h-4 w-4 text-blue-400 flex-shrink-0" />
        <span className="text-xs text-zinc-400">Current Time</span>
        {isRunning ? (
          <Loader2 className="h-3.5 w-3.5 text-violet-400 animate-spin ml-auto" />
        ) : result !== undefined ? (
          <span className="text-sm font-medium text-zinc-200 ml-auto">
            {result}
          </span>
        ) : null}
      </div>
    );
  },
});

/**
 * Tavily Search tool — shows query and search results
 */
export const SearchToolUI = makeAssistantToolUI<{ query: string }, string>({
  toolName: "tavily_search",
  render: ({ args, result, status }) => {
    const isRunning = status.type === "running";

    let parsedResults: Array<{
      title?: string;
      url?: string;
      content?: string;
    }> = [];
    if (result) {
      try {
        const parsed = JSON.parse(result);
        parsedResults = Array.isArray(parsed) ? parsed : [];
      } catch {
        // result is plain text
      }
    }

    return (
      <div className="my-1.5 rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2">
          {isRunning ? (
            <Loader2 className="h-4 w-4 text-violet-400 animate-spin flex-shrink-0" />
          ) : (
            <Search className="h-4 w-4 text-amber-400 flex-shrink-0" />
          )}
          <span className="text-xs text-zinc-400">Searching:</span>
          <span className="text-xs text-zinc-200 font-medium truncate">
            "{args.query}"
          </span>
        </div>

        {parsedResults.length > 0 && (
          <div className="border-t border-zinc-800 px-3 py-2 space-y-1.5">
            {parsedResults.slice(0, 4).map((item, i) => (
              <div
                key={i}
                className="flex items-start gap-2 text-xs"
              >
                <ExternalLink className="h-3 w-3 text-zinc-600 mt-0.5 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-zinc-300 font-medium truncate">
                    {item.title || item.url || "Result"}
                  </p>
                  {item.content && (
                    <p className="text-zinc-500 line-clamp-2">
                      {item.content.slice(0, 150)}
                    </p>
                  )}
                </div>
              </div>
            ))}
            {parsedResults.length > 4 && (
              <p className="text-[10px] text-zinc-600 pl-5">
                +{parsedResults.length - 4} more results
              </p>
            )}
          </div>
        )}

        {/* Show raw result if not parseable as array */}
        {result && parsedResults.length === 0 && !isRunning && (
          <div className="border-t border-zinc-800 px-3 py-2">
            <pre className="text-xs text-zinc-400 whitespace-pre-wrap break-all max-h-[150px] overflow-y-auto">
              {result.slice(0, 500)}
            </pre>
          </div>
        )}
      </div>
    );
  },
});

/**
 * Create Plan tool — shows structured task plan
 */
export const PlanToolUI = makeAssistantToolUI<
  { task_description: string },
  string
>({
  toolName: "create_plan",
  render: ({ args, result, status }) => {
    const isRunning = status.type === "running";

    let plan: { title?: string; tasks?: Array<{ title: string; done?: boolean }> } | null = null;
    if (result) {
      try {
        plan = JSON.parse(result);
      } catch {
        // not parseable
      }
    }

    return (
      <div className="my-1.5 rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2">
          {isRunning ? (
            <Loader2 className="h-4 w-4 text-violet-400 animate-spin flex-shrink-0" />
          ) : (
            <ListTodo className="h-4 w-4 text-violet-400 flex-shrink-0" />
          )}
          <span className="text-xs text-zinc-300 font-medium">
            {isRunning ? "Creating plan..." : plan?.title || "Task Plan"}
          </span>
        </div>

        {plan?.tasks && plan.tasks.length > 0 && (
          <div className="border-t border-zinc-800 px-3 py-2 space-y-1">
            {plan.tasks.map((task, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                {task.done ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                ) : (
                  <Circle className="h-3.5 w-3.5 text-zinc-600 mt-0.5 flex-shrink-0" />
                )}
                <span className="text-zinc-300">{task.title}</span>
              </div>
            ))}
          </div>
        )}

        {/* Fallback: show raw result if can't parse as plan */}
        {result && !plan?.tasks && !isRunning && (
          <div className="border-t border-zinc-800 px-3 py-2">
            <pre className="text-xs text-zinc-400 whitespace-pre-wrap break-all max-h-[150px] overflow-y-auto">
              {result.slice(0, 500)}
            </pre>
          </div>
        )}
      </div>
    );
  },
});
