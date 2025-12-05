import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useStore } from "../../lib/store";
import { socketService } from "../../infrastructure/websocket/websocket.service";
import { ApiService } from "../../infrastructure/api/api.service";
import ChatWindow from "../components/chat/chat.window";
import ChatInput from "../components/chat/chat.input";
import ResultsPanel from "../components/chat/results.panel";
import AgentProgress from "../components/chat/agent.progress";
import { Button } from "../components/ui/button";
import { ArrowLeft, CirclePause } from "lucide-react";
import toast from "react-hot-toast";
import type { AgentEvent, ChatMessage, HitlRequest } from "../../lib/types";

export default function ChatPage() {
  const { sessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();
  const {
    isConnected,
    setConnected,
    currentSessionId,
    setCurrentSession,
    setMessages,
    messages,
    addMessage,
    setAgentEvents,
    addAgentEvent,
    setPendingHitl,
    pendingHitl,
    isAnalyzing,
    setAnalyzing,
    resetSession,
  } = useStore();

  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const initialConnectionRef = useRef(false);

  useEffect(() => {
    if (!initialConnectionRef.current) {
      initialConnectionRef.current = true;
      initializeConnection();
    }

    return () => {
      socketService.off("session_created");
      socketService.off("agent_update");
      socketService.off("hitl_request");
      socketService.off("analysis_complete");
      socketService.off("analysis_failed");
      socketService.off("error");
    };
  }, []);
  useEffect(() => {
    if (!sessionId) {
      resetSession();
      setAnalysisResult(null);
    }
  }, [sessionId, resetSession]);

  const initializeConnection = async () => {
    try {
      const sid = await socketService.connect();
      setConnected(true, sid);
      console.log("[v0] Connected with socket ID:", sid);
      toast.success("Connected to server");

      registerSocketHandlers();
    } catch (error) {
      console.error("[v0] Connection failed:", error);
      setConnected(false);
      toast.error("Failed to connect to server");
    }
  };

  const registerSocketHandlers = () => {
    socketService.onSessionCreated((data) => {
      console.log("[v0] Session created:", data);
      setCurrentSession(data.session_id);
      addMessage({
        id: crypto.randomUUID(),
        type: "system",
        content: "Analysis session created. Starting workflow...",
        timestamp: new Date(),
      });
    });

    socketService.onAgentUpdate((data) => {
      addAgentEvent(data);

      if (data.status === "started") {
        setAnalyzing(true);
        addMessage({
          id: crypto.randomUUID(),
          type: "agent",
          content:
            data.payload.message || `${data.agent_name} is processing...`,
          timestamp: new Date(),
          agent_name: data.agent_name,
          metadata: data.payload,
        });
      } else if (data.status === "completed") {
        const requiresReview = Boolean(data.payload?.requires_review);
        const messageContent = requiresReview
          ? `${data.agent_name} output ready for review`
          : data.payload.message || `${data.agent_name} completed successfully`;

        const metadata = requiresReview
          ? { ...(data.payload ?? {}), requires_review: true }
          : data.payload;

        addMessage({
          id: crypto.randomUUID(),
          type: requiresReview ? "system" : "agent",
          content: messageContent,
          timestamp: new Date(),
          agent_name: requiresReview ? undefined : data.agent_name,
          metadata,
        });
      } else if (data.status === "failed") {
        addMessage({
          id: crypto.randomUUID(),
          type: "system",
          content: `${data.agent_name} failed: ${data.payload.message}`,
          timestamp: new Date(),
          metadata: data.payload,
        });
        toast.error(`${data.agent_name} failed`);
      } else if (data.status === "resolved") {
        useStore.setState((state) => ({
          messages: state.messages.map((message) => {
            if (
              message.type === "hitl" &&
              message.metadata?.session_id === data.session_id &&
              message.metadata?.stage === data.stage &&
              message.metadata?.resolved !== true
            ) {
              return {
                ...message,
                metadata: {
                  ...message.metadata,
                  resolved: true,
                  resolved_action: data.payload?.action,
                  resolved_at: new Date().toISOString(),
                },
              };
            }
            return message;
          }),
        }));

        const currentPending = useStore.getState().pendingHitl;
        if (currentPending && currentPending.stage === data.stage) {
          setPendingHitl(null);
        }

        const action = data.payload?.action;
        const actionLabel =
          typeof action === "string"
            ? {
                approve: "approved",
                deny: "denied",
                edit: "edited",
                recommend: "recommendation noted",
              }[action] || action
            : "approval resolved";

        addMessage({
          id: crypto.randomUUID(),
          type: "system",
          content: `${data.agent_name} ${actionLabel}`,
          timestamp: new Date(),
          metadata: data.payload,
        });

        const shouldResume =
          action === "approve" || action === "edit" || action === "recommend";
        setAnalyzing(shouldResume);
      }
    });

    socketService.onHitlRequest((data) => {
      console.log("[v0] HITL request:", data);

      const existingMessages = useStore
        .getState()
        .messages.filter(
          (msg) =>
            !(
              msg.type === "hitl" &&
              msg.metadata?.session_id === data.session_id &&
              msg.metadata?.stage === data.stage &&
              msg.metadata?.resolved !== true
            ),
        );
      setMessages(existingMessages);

      setPendingHitl(data);
      setAnalyzing(false);

      addMessage({
        id: crypto.randomUUID(),
        type: "hitl",
        content: `${data.agent_name} requires your approval`,
        timestamp: new Date(),
        agent_name: data.agent_name,
        metadata: {
          output: data.output,
          reasoning: data.reasoning,
          stage: data.stage,
          session_id: data.session_id,
          resolved: false,
        },
      });

      toast("Approval required", { icon: <CirclePause /> });
    });

    socketService.onAnalysisComplete((data) => {
      setAnalyzing(false);
      setAnalysisResult(data.result);

      addMessage({
        id: crypto.randomUUID(),
        type: "system",
        content: "Analysis completed successfully!",
        timestamp: new Date(),
        metadata: data.result,
      });

      toast.success("Analysis completed!");
    });

    socketService.onAnalysisFailed((data) => {
      console.log("[v0] Analysis failed:", data);
      setAnalyzing(false);

      const errorMsg = data.errors.join(", ");
      addMessage({
        id: crypto.randomUUID(),
        type: "system",
        content: `Analysis failed: ${errorMsg}`,
        timestamp: new Date(),
      });

      toast.error("Analysis failed");
    });

    socketService.onError((data) => {
      console.error("[v0] Socket error:", data);
      toast.error(data.message || "An error occurred");
    });
  };

  const loadExistingSession = useCallback(async (sid: string) => {
    setIsLoadingSession(true);
    try {
      resetSession();
      setAnalysisResult(null);
      const [session, rawEvents] = await Promise.all([
        ApiService.getSession(sid),
        ApiService.getSessionEvents(sid).catch(() => [] as AgentEvent[]),
      ]);

      setCurrentSession(sid);

      const eventHistory = [...rawEvents].sort((a, b) => {
        const aTime = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const bTime = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return aTime - bTime;
      });

      setAgentEvents(eventHistory);

      const historyMessages: ChatMessage[] = [
        {
          id: crypto.randomUUID(),
          type: "user",
          content: session.user_input,
          timestamp: new Date(session.created_at),
        },
      ];

      const pendingByStage = new Map<string, { request: HitlRequest; messageId: string }>();
      let lastResolvedAction: string | null = null;

      eventHistory.forEach((event) => {
        const eventTimestamp = event.timestamp
          ? new Date(event.timestamp)
          : new Date();
        const payload = event.payload ?? {};

        if (event.status === "started") {
          historyMessages.push({
            id: crypto.randomUUID(),
            type: "agent",
            content:
              payload.message || `${event.agent_name} is processing...`,
            timestamp: eventTimestamp,
            agent_name: event.agent_name,
            metadata: payload,
          });
          return;
        }

        if (event.status === "completed") {
          const requiresReview = Boolean(payload?.requires_review);
          const messageContent = requiresReview
            ? `${event.agent_name} output ready for review`
            : payload?.message || `${event.agent_name} completed successfully`;

          const metadata = requiresReview
            ? { ...(payload ?? {}), requires_review: true }
            : payload;

          historyMessages.push({
            id: crypto.randomUUID(),
            type: requiresReview ? "system" : "agent",
            content: messageContent,
            timestamp: eventTimestamp,
            agent_name: requiresReview ? undefined : event.agent_name,
            metadata,
          });
          return;
        }

        if (event.status === "failed") {
          historyMessages.push({
            id: crypto.randomUUID(),
            type: "system",
            content: `${event.agent_name} failed: ${payload.message ?? "See details"}`,
            timestamp: eventTimestamp,
            metadata: payload,
          });
          return;
        }

        if (event.status === "pending") {
          const hitlMetadata = {
            ...payload,
            stage: event.stage,
            session_id: event.session_id,
            resolved: false,
          };

          const messageId = crypto.randomUUID();

          historyMessages.push({
            id: messageId,
            type: "hitl",
            content: `${event.agent_name} requires your approval`,
            timestamp: eventTimestamp,
            agent_name: event.agent_name,
            metadata: hitlMetadata,
          });

          pendingByStage.set(event.stage, {
            messageId,
            request: {
              session_id: event.session_id,
              stage: event.stage,
              agent_name: event.agent_name,
              output:
                payload.output ??
                payload.output_preview ??
                payload ??
                null,
              reasoning:
                payload.reasoning ??
                payload.message ??
                (typeof payload === "string" ? payload : ""),
            },
          });
          return;
        }

        if (event.status === "resolved") {
          const pendingEntry = pendingByStage.get(event.stage);
          if (pendingEntry) {
            const index = historyMessages.findIndex(
              (message) => message.id === pendingEntry.messageId,
            );
            if (index !== -1) {
              const existing = historyMessages[index];
              historyMessages[index] = {
                ...existing,
                metadata: {
                  ...(existing.metadata ?? {}),
                  resolved: true,
                  resolved_action: payload?.action,
                },
              };
            }
          }

          pendingByStage.delete(event.stage);

          const action = payload?.action;
          const actionLabel =
            typeof action === "string"
              ? {
                  approve: "approved",
                  deny: "denied",
                  edit: "edited",
                  recommend: "recommendation noted",
                }[action] || action
              : "approval resolved";

          if (typeof action === "string") {
            lastResolvedAction = action;
          }

          historyMessages.push({
            id: crypto.randomUUID(),
            type: "system",
            content: `${event.agent_name} ${actionLabel}`,
            timestamp: eventTimestamp,
            metadata: payload,
          });
        }
      });

      let analysisResult: any = null;
      if (session.status === "completed") {
        try {
          analysisResult = await ApiService.getResults(sid);
          setAnalysisResult(analysisResult);

          historyMessages.push({
            id: crypto.randomUUID(),
            type: "system",
            content: "Loaded previous analysis results",
            timestamp: analysisResult?.created_at
              ? new Date(analysisResult.created_at)
              : new Date(session.updated_at),
            metadata: analysisResult,
          });
        } catch (error) {
          console.error("[v0] Failed to load results:", error);
        }
      } else {
        historyMessages.push({
          id: crypto.randomUUID(),
          type: "system",
          content: `Continuando análisis en etapa ${session.current_stage}`,
          timestamp: new Date(session.updated_at || Date.now()),
        });
      }

      historyMessages.sort(
        (a, b) => a.timestamp.getTime() - b.timestamp.getTime(),
      );

      setMessages(historyMessages);

      const pendingFromHistoryEntry =
        pendingByStage.size > 0
          ? Array.from(pendingByStage.values()).pop() ?? null
          : null;

      setPendingHitl(pendingFromHistoryEntry?.request ?? null);
      const shouldAnalyze =
        session.status === "processing" &&
        !pendingFromHistoryEntry &&
        lastResolvedAction !== "deny";
      setAnalyzing(shouldAnalyze);

      toast.success("Session loaded");
    } catch (error) {
      console.error("[v0] Failed to load session:", error);
      toast.error("Failed to load session");
    } finally {
      setIsLoadingSession(false);
    }
  }, [
    resetSession,
    setAnalysisResult,
    setCurrentSession,
    setAgentEvents,
    setMessages,
    setPendingHitl,
    setAnalyzing,
  ]);

  useEffect(() => {
    if (sessionId && !currentSessionId) {
      loadExistingSession(sessionId);
    }
  }, [sessionId, currentSessionId, loadExistingSession]);

  const handleSubmitAnalysis = (userInput: string) => {
    if (!isConnected) {
      toast.error("Not connected to server");
      return;
    }

    if (currentSessionId) {
      resetSession();
      setAnalysisResult(null);
    }

    addMessage({
      id: crypto.randomUUID(),
      type: "user",
      content: userInput,
      timestamp: new Date(),
    });

    setAnalyzing(true);
    socketService.startAnalysis({
      user_input: userInput,
    });

    toast("Starting analysis...", { icon: "🚀" });
  };

  const handleHitlApprove = () => {
    if (!pendingHitl || !currentSessionId) return;

    socketService.respondToHitl({
      session_id: currentSessionId,
      action: "approve",
      stage: pendingHitl.stage,
    });

    setPendingHitl(null);
    setAnalyzing(true);
    addMessage({
      id: crypto.randomUUID(),
      type: "system",
      content: `Confirmaste la salida de ${pendingHitl.agent_name}`,
      timestamp: new Date(),
    });
    toast.success("Approved! Continuing...");
  };

  const handleHitlDeny = (feedback: string) => {
    if (!pendingHitl || !currentSessionId) return;

    socketService.respondToHitl({
      session_id: currentSessionId,
      action: "deny",
      feedback,
      stage: pendingHitl.stage,
    });

    setPendingHitl(null);
    setAnalyzing(true);
    addMessage({
      id: crypto.randomUUID(),
      type: "system",
      content: `Rechazaste la salida de ${pendingHitl.agent_name}`,
      timestamp: new Date(),
    });
    toast("Feedback enviado. Solicitando nueva iteración...", {
      icon: "♻️",
    });
  };

  const handleHitlEdit = (editedOutput: any) => {
    if (!pendingHitl || !currentSessionId) return;

    socketService.respondToHitl({
      session_id: currentSessionId,
      action: "edit",
      edited_output: editedOutput,
      stage: pendingHitl.stage,
    });

    setPendingHitl(null);
    setAnalyzing(true);
    addMessage({
      id: crypto.randomUUID(),
      type: "system",
      content: `Editaste la salida de ${pendingHitl.agent_name}`,
      timestamp: new Date(),
    });
    toast.success("Edited! Continuing with changes...");
  };

  const handleHitlRecommend = (recommendation: string) => {
    if (!pendingHitl || !currentSessionId) return;
    if (!recommendation.trim()) {
      toast.error("Please provide a recommendation");
      return;
    }

    socketService.respondToHitl({
      session_id: currentSessionId,
      action: "recommend",
      feedback: recommendation,
      stage: pendingHitl.stage,
    });

    setPendingHitl(null);
    setAnalyzing(true);
    addMessage({
      id: crypto.randomUUID(),
      type: "system",
      content: `Enviaste recomendaciones para ${pendingHitl.agent_name}`,
      timestamp: new Date(),
    });
    toast.success("Recommendation sent. Continuing...");
  };

  const handleBack = () => {
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="bg-card py-3 flex-shrink-0 sticky top-0 z-30 backdrop-blur px-[4%]">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={handleBack}
              className="rounded-full"
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <h1 className="text-xl font-semibold text-foreground">
              {currentSessionId ? "Analysis Session" : "New Analysis"}
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`}
            />
            <span className="text-sm text-muted-foreground">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
        <AgentProgress hasActiveSession={Boolean(currentSessionId)} />
      </header>

      <div className="flex-1 overflow-hidden flex">
        <div className="flex-1 flex flex-col">
          <ChatWindow
            messages={messages}
            isAnalyzing={isAnalyzing}
            isLoadingSession={isLoadingSession}
            pendingHitl={pendingHitl}
            onHitlApprove={handleHitlApprove}
            onHitlDeny={handleHitlDeny}
            onHitlEdit={handleHitlEdit}
            onHitlRecommend={handleHitlRecommend}
          />

          {!analysisResult && (
            <ChatInput
              onSubmit={handleSubmitAnalysis}
              disabled={isAnalyzing || !isConnected}
            />
          )}
        </div>

        {analysisResult && (
          <div className="w-full lg:w-2/5 border-l border-border overflow-y-auto">
            <ResultsPanel result={analysisResult} />
          </div>
        )}
      </div>

    </div>
  );
}
