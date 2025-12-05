import type { IApiService } from "../../domain/interfaces/iapi.service";
import { environment } from "../../environments/environment";
import type {
  Session,
  AnalysisResult,
  HitlResponse,
  AgentEvent,
} from "../../lib/types";
import axios from "axios";

const API_BASE_URL = environment.BASE_URL;

export class ApiService implements IApiService {
  static async checkHealth(): Promise<{ status: string; database: string }> {
    const response = await axios.get(`${API_BASE_URL}/health`);
    if (response.status !== 200) {
      throw new Error("Health check failed");
    }
    return response.data;
  }

  static async createSession(
    userInput: string,
    socketId?: string,
  ): Promise<{ session_id: string; status: string }> {
    const payload: any = {
      user_input: userInput,
    };
    if (socketId) {
      payload.socket_id = socketId;
    }

    const response = await axios.post(`${API_BASE_URL}/sessions`, payload);

    if (response.status !== 201) {
      throw new Error("Failed to create session");
    }
    return response.data;
  }

  static async getSession(sessionId: string): Promise<Session> {
    const response = await axios.get(`${API_BASE_URL}/sessions/${sessionId}`);

    if (response.status !== 200) {
      throw new Error("Failed to fetch session");
    }
    return response.data;
  }

  static async listSessions(limit = 12): Promise<Session[]> {
    const response = await axios.get(`${API_BASE_URL}/sessions?limit=${limit}`);
    if (response.status !== 200) {
      throw new Error("Failed to fetch recent sessions");
    }
    const data = response.data;
    return data.sessions || [];
  }

  static async getResults(sessionId: string): Promise<AnalysisResult> {
    const response = await axios.get(
      `${API_BASE_URL}/sessions/${sessionId}/results`,
    );

    if (response.status !== 200) {
      throw new Error("Failed to fetch results");
    }
    return response.data;
  }

  static async getSessionEvents(sessionId: string): Promise<AgentEvent[]> {
    const response = await axios.get(
      `${API_BASE_URL}/sessions/${sessionId}/events`,
    );

    if (response.status !== 200) {
      throw new Error("Failed to fetch session events");
    }

    return response.data?.events ?? [];
  }

  static async submitHitlResponse(
    sessionId: string,
    responsePayload: HitlResponse,
  ): Promise<void> {
    const response = await axios.post(
      `${API_BASE_URL}/sessions/${sessionId}/hitl`,
      responsePayload,
    );

    if (response.status !== 200) {
      throw new Error("Failed to submit HITL response");
    }
  }
}
