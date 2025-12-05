import type {
  StartAnalysisPayload,
  HitlResponse,
  SessionCreatedPayload,
  AgentEvent,
  HitlRequest,
  AnalysisCompletePayload,
  AnalysisFailedPayload,
  ErrorPayload,
} from "../../lib/types";

export type ConnectionCallback = (socketId: string) => void;
export type SessionCreatedCallback = (data: SessionCreatedPayload) => void;
export type AgentUpdateCallback = (data: AgentEvent) => void;
export type HitlRequestCallback = (data: HitlRequest) => void;
export type AnalysisCompleteCallback = (data: AnalysisCompletePayload) => void;
export type AnalysisFailedCallback = (data: AnalysisFailedPayload) => void;
export type ErrorCallback = (data: ErrorPayload) => void;

export interface ISocketService {
  connect(): Promise<string>;
  disconnect(): void;
  isConnected(): boolean;
  startAnalysis(payload: StartAnalysisPayload): void;
  respondToHitl(response: HitlResponse): void;
  onConnectionEstablished(callback: ConnectionCallback): void;
  onSessionCreated(callback: SessionCreatedCallback): void;
  onAgentUpdate(callback: AgentUpdateCallback): void;
  onHitlRequest(callback: HitlRequestCallback): void;
  onAnalysisComplete(callback: AnalysisCompleteCallback): void;
  onAnalysisFailed(callback: AnalysisFailedCallback): void;
  onError(callback: ErrorCallback): void;
  off(event: string, callback?: (...args: any[]) => void): void;
}
