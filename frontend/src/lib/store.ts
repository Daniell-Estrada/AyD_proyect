import { create } from "zustand";
import type { ChatMessage, HitlRequest, AgentEvent } from "./types";

interface AppState {
  isConnected: boolean;
  socketId: string | null;

  currentSessionId: string | null;
  messages: ChatMessage[];
  agentEvents: AgentEvent[];

  pendingHitl: HitlRequest | null;
  isAnalyzing: boolean;

  setConnected: (connected: boolean, socketId?: string) => void;
  setCurrentSession: (sessionId: string | null) => void;
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  setAgentEvents: (events: AgentEvent[]) => void;
  addAgentEvent: (event: AgentEvent) => void;
  setPendingHitl: (hitl: HitlRequest | null) => void;
  setAnalyzing: (analyzing: boolean) => void;
  resetSession: () => void;
}

export const useStore = create<AppState>((set) => ({
  isConnected: false,
  socketId: null,
  currentSessionId: null,
  messages: [],
  agentEvents: [],
  pendingHitl: null,
  isAnalyzing: false,

  setConnected: (connected, socketId) =>
    set({ isConnected: connected, socketId: socketId || null }),

  setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),

  setMessages: (messages) => set({ messages }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  setAgentEvents: (events) => set({ agentEvents: events }),

  addAgentEvent: (event) =>
    set((state) => ({ agentEvents: [...state.agentEvents, event] })),

  setPendingHitl: (hitl) => set({ pendingHitl: hitl }),

  setAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),

  resetSession: () =>
    set({
      currentSessionId: null,
      messages: [],
      agentEvents: [],
      pendingHitl: null,
      isAnalyzing: false,
    }),
}));
