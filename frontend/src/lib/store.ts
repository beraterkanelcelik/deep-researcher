import { create } from "zustand";

interface AppState {
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  currentThreadId: string | null;
  setCurrentThreadId: (id: string | null) => void;

  threads: Array<{ thread_id: string; title: string; updated_at: string }>;
  setThreads: (
    threads: Array<{ thread_id: string; title: string; updated_at: string }>
  ) => void;

  selectedModel: string;
  setSelectedModel: (model: string) => void;
}

export type NodeStatus = "pending" | "active" | "completed";

export interface NodeState {
  name: string;
  status: NodeStatus;
  runCount: number;
}

export interface SubgraphNodeState {
  name: string;
  status: NodeStatus;
  runCount: number;
}

// HITL types matching backend schemas
export interface HITLOption {
  id: string;
  label: string;
  description: string;
  selected: boolean;
}

export interface ResearchFinding {
  insight: string;
  evidence: string;
  sources: string[];
}

export interface ResearchReport {
  title: string;
  summary: string;
  key_findings: ResearchFinding[];
  sources: string[];
  tags: string[];
  methodology: string;
}

export interface HITLPayload {
  hitl_type: "checkbox" | "yes_no" | "select" | "text" | "review" | "confirm";
  title: string;
  message: string;
  options: HITLOption[];
  report: ResearchReport | null;
}

interface NodeStatusState {
  nodes: NodeState[];
  subgraphNodes: SubgraphNodeState[];
  isGraphRunning: boolean;
  isSubgraphRunning: boolean;
  activeInterrupt: HITLPayload | null;
  handleNodeEvent: (
    node: string,
    status: string,
    isSubgraph?: boolean
  ) => void;
  setInterrupt: (payload: HITLPayload | null) => void;
  reset: () => void;
}

export const useNodeStatusStore = create<NodeStatusState>((set) => ({
  nodes: [],
  subgraphNodes: [],
  isGraphRunning: false,
  isSubgraphRunning: false,
  activeInterrupt: null,

  handleNodeEvent: (node, status, isSubgraph) => {
    if (node === "__graph__") {
      if (status === "started") {
        set({
          isGraphRunning: true,
          nodes: [],
          subgraphNodes: [],
          isSubgraphRunning: false,
        });
      } else if (status === "finished") {
        set({ isGraphRunning: false, isSubgraphRunning: false });
      }
      return;
    }

    // Handle subgraph nodes
    if (isSubgraph) {
      set((state) => {
        const exists = state.subgraphNodes.some((n) => n.name === node);

        let newSubgraphNodes: SubgraphNodeState[];
        if (!exists) {
          // Dynamically add new subgraph node
          newSubgraphNodes = [
            ...state.subgraphNodes,
            {
              name: node,
              status: (status === "active" || status === "completed" ? status : "active") as NodeStatus,
              runCount: status === "active" ? 1 : 0,
            },
          ];
        } else {
          newSubgraphNodes = state.subgraphNodes.map((n) => {
            if (n.name !== node) return n;
            if (status === "active") {
              return { ...n, status: "active" as NodeStatus, runCount: n.runCount + 1 };
            }
            if (status === "completed") {
              return { ...n, status: "completed" as NodeStatus };
            }
            return n;
          });
        }

        const anyActive = newSubgraphNodes.some((n) => n.status === "active");
        const anyCompleted = newSubgraphNodes.some(
          (n) => n.status === "completed" || n.status === "active"
        );

        return {
          subgraphNodes: newSubgraphNodes,
          isSubgraphRunning: anyActive || anyCompleted,
        };
      });
      return;
    }

    // Handle main graph nodes
    set((state) => {
      const exists = state.nodes.some((n) => n.name === node);

      if (!exists) {
        // Dynamically add new main graph node
        return {
          nodes: [
            ...state.nodes,
            {
              name: node,
              status: (status === "active" || status === "completed" ? status : "active") as NodeStatus,
              runCount: status === "active" ? 1 : 0,
            },
          ],
        };
      }

      return {
        nodes: state.nodes.map((n) => {
          if (n.name !== node) return n;
          if (status === "active") {
            return { ...n, status: "active" as NodeStatus, runCount: n.runCount + 1 };
          }
          if (status === "completed") {
            return { ...n, status: "completed" as NodeStatus };
          }
          return n;
        }),
      };
    });
  },

  setInterrupt: (payload) => set({ activeInterrupt: payload }),

  reset: () =>
    set({
      nodes: [],
      subgraphNodes: [],
      isGraphRunning: false,
      isSubgraphRunning: false,
      activeInterrupt: null,
    }),
}));

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  currentThreadId: null,
  setCurrentThreadId: (id) => set({ currentThreadId: id }),

  threads: [],
  setThreads: (threads) => set({ threads }),

  selectedModel: "gpt-4.1-mini",
  setSelectedModel: (model) => set({ selectedModel: model }),
}));
