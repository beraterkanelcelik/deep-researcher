import {
  Search,
  Bot,
  Wrench,
  Check,
  Microscope,
  ListTodo,
  Compass,
  Eye,
  FileCheck,
  Save,
  HelpCircle,
  Circle,
} from "lucide-react";
import {
  useNodeStatusStore,
  type NodeState,
  type SubgraphNodeState,
} from "@/lib/store";

// Known node configs for nice icons/labels — unknown nodes get a fallback
const knownNodeConfig: Record<
  string,
  { label: string; icon: React.ComponentType<{ className?: string }> }
> = {
  retrieve: { label: "Retrieve", icon: Search },
  agent: { label: "Agent", icon: Bot },
  tools: { label: "Tools", icon: Wrench },
  deep_research: { label: "Research", icon: Microscope },
  save_confirm: { label: "Confirm", icon: FileCheck },
  save_to_db: { label: "Save", icon: Save },
  clarify: { label: "Clarify", icon: HelpCircle },
  orchestrate: { label: "Plan", icon: ListTodo },
  explorer: { label: "Explore", icon: Compass },
  synthesize: { label: "Synthesize", icon: FileCheck },
  review: { label: "Review", icon: Eye },
};

function getNodeConfig(name: string) {
  if (knownNodeConfig[name]) return knownNodeConfig[name];
  // Fallback: capitalize the node name, use a generic icon
  const label = name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return { label, icon: Circle };
}

function NodeStep({ node }: { node: NodeState | SubgraphNodeState }) {
  const config = getNodeConfig(node.name);
  const Icon = config.icon;

  const isActive = node.status === "active";
  const isCompleted = node.status === "completed";

  return (
    <div
      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300 ${
        isActive
          ? "text-violet-400 bg-violet-500/10"
          : isCompleted
            ? "text-emerald-400 bg-emerald-500/10"
            : "text-zinc-600"
      }`}
    >
      <div className="relative flex-shrink-0">
        {isCompleted ? (
          <Check className="h-3.5 w-3.5" />
        ) : (
          <>
            <Icon className="h-3.5 w-3.5" />
            {isActive && (
              <span className="absolute -top-0.5 -right-0.5 h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse" />
            )}
          </>
        )}
      </div>
      <span>{config.label}</span>
      {node.runCount > 1 && (
        <span className="text-[10px] opacity-70">x{node.runCount}</span>
      )}
    </div>
  );
}

function Connector({ completed }: { completed: boolean }) {
  return (
    <div className="flex justify-center py-0.5">
      <div
        className={`h-3 w-px transition-colors duration-300 ${
          completed ? "bg-emerald-500/40" : "bg-zinc-800"
        }`}
      />
    </div>
  );
}

export function GraphNodeIndicator() {
  const nodes = useNodeStatusStore((s) => s.nodes);
  const subgraphNodes = useNodeStatusStore((s) => s.subgraphNodes);
  const isGraphRunning = useNodeStatusStore((s) => s.isGraphRunning);
  const isSubgraphRunning = useNodeStatusStore((s) => s.isSubgraphRunning);

  // Don't render anything if no nodes have been seen yet
  if (nodes.length === 0 && subgraphNodes.length === 0) return null;

  return (
    <div className="fixed right-0 top-1/2 -translate-y-1/2 z-50">
      <div
        className={`flex flex-col border border-zinc-800 rounded-l-xl bg-zinc-900/90 backdrop-blur-sm px-2 py-3 transition-opacity duration-300 ${
          isGraphRunning ? "opacity-100" : "opacity-40"
        }`}
      >
        <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider px-3 pb-2">
          Pipeline
        </div>
        {nodes.map((node, i) => (
          <div key={node.name}>
            {i > 0 && (
              <Connector
                completed={nodes[i - 1].status === "completed"}
              />
            )}
            <NodeStep node={node} />
          </div>
        ))}

        {/* Deep Research subgraph section */}
        {isSubgraphRunning && subgraphNodes.length > 0 && (
          <div className="mt-3 pt-2 border-t border-zinc-800">
            <div className="text-[10px] font-semibold text-violet-500 uppercase tracking-wider px-3 pb-2 flex items-center gap-1.5">
              <Microscope className="h-3 w-3" />
              Subgraph
            </div>
            {subgraphNodes.map((node, i) => (
              <div key={node.name}>
                {i > 0 && (
                  <Connector
                    completed={
                      subgraphNodes[i - 1].status === "completed"
                    }
                  />
                )}
                <NodeStep node={node} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
