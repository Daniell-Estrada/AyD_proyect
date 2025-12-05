import { io, type Socket } from "socket.io-client";
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
import { environment } from "../../environments/environment";
import type { ISocketService } from "../../domain/interfaces/isocket.service";
import { useStore } from "../../lib/store";

class SocketService implements ISocketService {
  private socket: Socket | null = null;
  private readonly url = environment.WS_URL;

  connect(): Promise<string> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        const { setConnected } = useStore.getState();
        setConnected(true, this.socket.id || undefined);
        resolve(this.socket.id || "");
        return;
      }

      this.socket = io(this.url, {
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      const setConnected = () =>
        useStore.getState().setConnected(true, this.socket?.id);
      const setDisconnected = () => useStore.getState().setConnected(false);

      this.socket.on("connect", () => {
        console.log("[v0] Socket connected:", this.socket?.id);
        setConnected();
        resolve(this.socket?.id || "");
      });

      this.socket.on("connection_established", (data: { sid: string }) => {
        console.log("[v0] Connection established:", data.sid);
      });

      this.socket.on("connect_error", (error) => {
        console.error("[v0] Socket connection error:", error);
        setDisconnected();
        reject(error);
      });

      this.socket.on("disconnect", () => {
        console.log("[v0] Socket disconnected");
        setDisconnected();
      });

      this.socket.io.on("reconnect", () => {
        console.log("[v0] Socket reconnected:", this.socket?.id);
        setConnected();
      });

      this.socket.io.on("reconnect_attempt", () => {
        setDisconnected();
      });
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  startAnalysis(payload: StartAnalysisPayload) {
    if (!this.socket) {
      throw new Error("Socket not connected");
    }
    console.log("[v0] Starting analysis:", payload);
    this.socket.emit("start_analysis", payload);
  }

  respondToHitl(response: HitlResponse) {
    if (!this.socket) {
      throw new Error("Socket not connected");
    }
    console.log("[v0] HITL response:", response);
    this.socket.emit("hitl_respond", response);
  }

  onSessionCreated(callback: (data: SessionCreatedPayload) => void) {
    if (!this.socket) return;
    this.socket.on("session_created", callback);
  }

  onAgentUpdate(callback: (data: AgentEvent) => void) {
    if (!this.socket) return;
    this.socket.on("agent_update", callback);
  }

  onHitlRequest(callback: (data: HitlRequest) => void) {
    if (!this.socket) return;
    this.socket.on("hitl_request", callback);
  }

  onAnalysisComplete(callback: (data: AnalysisCompletePayload) => void) {
    if (!this.socket) return;
    this.socket.on("analysis_complete", callback);
  }

  onAnalysisFailed(callback: (data: AnalysisFailedPayload) => void) {
    if (!this.socket) return;
    this.socket.on("analysis_failed", callback);
  }

  onError(callback: (data: ErrorPayload) => void) {
    if (!this.socket) return;
    this.socket.on("error", callback);
  }

  off(event: string, callback?: (...args: any[]) => void) {
    if (!this.socket) return;
    this.socket.off(event, callback);
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

const socketService = new SocketService();
export { socketService };
