import type { Session, AnalysisResult, HitlResponse } from "../../lib/types";

export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface IApiService {
  checkHealth(): Promise<{ status: string; database: string }>;

  createSession(
    userInput: string,
    socketId?: string,
  ): Promise<{ session_id: string; status: string }>;

  getSession(sessionId: string): Promise<Session>;
  getResults(sessionId: string): Promise<AnalysisResult>;
  listSessions(limit?: number): Promise<Session[]>;
  submitHitlResponse(sessionId: string, response: HitlResponse): Promise<void>;
}
