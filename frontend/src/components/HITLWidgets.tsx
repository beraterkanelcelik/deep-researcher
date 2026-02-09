import { useState } from "react";
import {
  CheckSquare,
  Square,
  Send,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Save,
  X,
  Pencil,
  ExternalLink,
} from "lucide-react";
import type { HITLPayload, HITLOption, ResearchReport } from "@/lib/store";

interface HITLWidgetProps {
  payload: HITLPayload;
  onResume: (value: unknown) => void;
}

export function HITLWidget({ payload, onResume }: HITLWidgetProps) {
  switch (payload.hitl_type) {
    case "checkbox":
      return <CheckboxWidget payload={payload} onResume={onResume} />;
    case "yes_no":
      return <YesNoWidget payload={payload} onResume={onResume} />;
    case "select":
      return <SelectWidget payload={payload} onResume={onResume} />;
    case "text":
      return <TextWidget payload={payload} onResume={onResume} />;
    case "review":
      return <ReviewWidget payload={payload} onResume={onResume} />;
    case "confirm":
      return <ConfirmWidget payload={payload} onResume={onResume} />;
    default:
      return (
        <div className="text-zinc-400 text-sm p-4">
          Unknown HITL type: {payload.hitl_type}
        </div>
      );
  }
}

function WidgetContainer({
  title,
  message,
  children,
}: {
  title: string;
  message: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-2xl px-4 py-3">
      <div className="rounded-xl border border-violet-500/30 bg-zinc-900/80 backdrop-blur-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-zinc-800 bg-violet-500/5">
          <h3 className="text-sm font-semibold text-violet-300">{title}</h3>
          <p className="text-xs text-zinc-400 mt-1">{message}</p>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

function CheckboxWidget({ payload, onResume }: HITLWidgetProps) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(payload.options.filter((o) => o.selected).map((o) => o.id))
  );

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <WidgetContainer title={payload.title} message={payload.message}>
      <div className="space-y-2 mb-4">
        {payload.options.map((opt) => (
          <button
            key={opt.id}
            onClick={() => toggle(opt.id)}
            className="flex items-start gap-3 w-full text-left px-3 py-2 rounded-lg hover:bg-zinc-800 transition-colors"
          >
            {selected.has(opt.id) ? (
              <CheckSquare className="h-4 w-4 text-violet-400 mt-0.5 flex-shrink-0" />
            ) : (
              <Square className="h-4 w-4 text-zinc-600 mt-0.5 flex-shrink-0" />
            )}
            <div>
              <span className="text-sm text-zinc-200">{opt.label}</span>
              {opt.description && (
                <p className="text-xs text-zinc-500 mt-0.5">
                  {opt.description}
                </p>
              )}
            </div>
          </button>
        ))}
      </div>
      <button
        onClick={() => onResume(Array.from(selected))}
        disabled={selected.size === 0}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Send className="h-3.5 w-3.5" />
        Continue with {selected.size} selected
      </button>
    </WidgetContainer>
  );
}

function YesNoWidget({ payload, onResume }: HITLWidgetProps) {
  return (
    <WidgetContainer title={payload.title} message={payload.message}>
      <div className="flex gap-3">
        <button
          onClick={() => onResume({ action: "yes" })}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors"
        >
          <ThumbsUp className="h-3.5 w-3.5" />
          Yes
        </button>
        <button
          onClick={() => onResume({ action: "no" })}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 text-zinc-200 text-sm font-medium hover:bg-zinc-600 transition-colors"
        >
          <ThumbsDown className="h-3.5 w-3.5" />
          No
        </button>
      </div>
    </WidgetContainer>
  );
}

function SelectWidget({ payload, onResume }: HITLWidgetProps) {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <WidgetContainer title={payload.title} message={payload.message}>
      <div className="space-y-2 mb-4">
        {payload.options.map((opt) => (
          <button
            key={opt.id}
            onClick={() => setSelected(opt.id)}
            className={`flex items-center gap-3 w-full text-left px-3 py-2 rounded-lg transition-colors ${
              selected === opt.id
                ? "bg-violet-500/20 border border-violet-500/40"
                : "hover:bg-zinc-800 border border-transparent"
            }`}
          >
            <div
              className={`w-3.5 h-3.5 rounded-full border-2 flex-shrink-0 ${
                selected === opt.id
                  ? "border-violet-400 bg-violet-400"
                  : "border-zinc-600"
              }`}
            />
            <div>
              <span className="text-sm text-zinc-200">{opt.label}</span>
              {opt.description && (
                <p className="text-xs text-zinc-500 mt-0.5">
                  {opt.description}
                </p>
              )}
            </div>
          </button>
        ))}
      </div>
      <button
        onClick={() => selected && onResume({ action: selected })}
        disabled={!selected}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Send className="h-3.5 w-3.5" />
        Confirm
      </button>
    </WidgetContainer>
  );
}

function TextWidget({ payload, onResume }: HITLWidgetProps) {
  const [text, setText] = useState("");

  return (
    <WidgetContainer title={payload.title} message={payload.message}>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg p-3 text-sm text-zinc-200 placeholder:text-zinc-600 resize-none outline-none focus:border-violet-500/50 min-h-[80px] mb-3"
        placeholder="Type your response..."
      />
      <button
        onClick={() => onResume({ text })}
        disabled={!text.trim()}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Send className="h-3.5 w-3.5" />
        Submit
      </button>
    </WidgetContainer>
  );
}

function ReportPreview({ report }: { report: ResearchReport }) {
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-800/50 p-4 space-y-3">
      <h4 className="text-sm font-semibold text-zinc-200">{report.title}</h4>
      <p className="text-xs text-zinc-400 leading-relaxed">{report.summary}</p>

      {report.key_findings.length > 0 && (
        <div>
          <h5 className="text-xs font-medium text-zinc-300 mb-1.5">
            Key Findings
          </h5>
          <ul className="space-y-1.5">
            {report.key_findings.map((finding, i) => (
              <li key={i} className="text-xs text-zinc-400">
                <span className="text-violet-400 font-medium">
                  {i + 1}.
                </span>{" "}
                {finding.insight}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.sources.length > 0 && (
        <div>
          <h5 className="text-xs font-medium text-zinc-300 mb-1">Sources</h5>
          <div className="flex flex-wrap gap-1.5">
            {report.sources.slice(0, 5).map((url, i) => (
              <a
                key={i}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[10px] text-violet-400 hover:text-violet-300 bg-violet-500/10 px-2 py-0.5 rounded"
              >
                <ExternalLink className="h-2.5 w-2.5" />
                Source {i + 1}
              </a>
            ))}
            {report.sources.length > 5 && (
              <span className="text-[10px] text-zinc-500 px-2 py-0.5">
                +{report.sources.length - 5} more
              </span>
            )}
          </div>
        </div>
      )}

      {report.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {report.tags.map((tag, i) => (
            <span
              key={i}
              className="text-[10px] text-zinc-400 bg-zinc-700/50 px-2 py-0.5 rounded"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ReviewWidget({ payload, onResume }: HITLWidgetProps) {
  const [mode, setMode] = useState<"view" | "edit">("view");
  const [editedSummary, setEditedSummary] = useState(
    payload.report?.summary || ""
  );

  return (
    <WidgetContainer title={payload.title} message={payload.message}>
      {payload.report && <ReportPreview report={payload.report} />}

      {mode === "edit" && (
        <div className="mt-3">
          <label className="text-xs text-zinc-400 mb-1 block">
            Edit Summary
          </label>
          <textarea
            value={editedSummary}
            onChange={(e) => setEditedSummary(e.target.value)}
            className="w-full bg-zinc-800 border border-zinc-700 rounded-lg p-3 text-sm text-zinc-200 resize-none outline-none focus:border-violet-500/50 min-h-[80px]"
          />
        </div>
      )}

      <div className="flex gap-2 mt-4">
        <button
          onClick={() =>
            onResume(
              mode === "edit"
                ? { action: "edit", edits: { summary: editedSummary } }
                : { action: "approve" }
            )
          }
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors"
        >
          <ThumbsUp className="h-3.5 w-3.5" />
          {mode === "edit" ? "Save & Approve" : "Approve"}
        </button>
        {mode === "view" && (
          <button
            onClick={() => setMode("edit")}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 text-zinc-200 text-sm font-medium hover:bg-zinc-600 transition-colors"
          >
            <Pencil className="h-3.5 w-3.5" />
            Edit
          </button>
        )}
        <button
          onClick={() => onResume({ action: "redo" })}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 text-zinc-200 text-sm font-medium hover:bg-zinc-600 transition-colors"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Redo
        </button>
      </div>
    </WidgetContainer>
  );
}

function ConfirmWidget({ payload, onResume }: HITLWidgetProps) {
  return (
    <WidgetContainer title={payload.title} message={payload.message}>
      {payload.report && <ReportPreview report={payload.report} />}

      <div className="flex gap-2 mt-4">
        <button
          onClick={() => onResume({ action: "save" })}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors"
        >
          <Save className="h-3.5 w-3.5" />
          Save to Database
        </button>
        <button
          onClick={() => onResume({ action: "cancel" })}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-700 text-zinc-200 text-sm font-medium hover:bg-zinc-600 transition-colors"
        >
          <X className="h-3.5 w-3.5" />
          Cancel
        </button>
      </div>
    </WidgetContainer>
  );
}
