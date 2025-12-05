import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiService } from "../../infrastructure/api/api.service";
import type { Session } from "../../lib/types";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Code, Brain, Sparkles } from "lucide-react";
import toast from "react-hot-toast";
import { useStore } from "../../lib/store";

export default function HomePage() {
  const navigate = useNavigate();
  const resetSession = useStore((state) => state.resetSession);
  const isSocketConnected = useStore((state) => state.isConnected);
  const [recentSessions, setRecentSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  type HealthStatus = "healthy" | "error" | "checking";
  const [healthStatus, setHealthStatus] = useState<HealthStatus>("checking");
  const errorToastShownRef = useRef(false);

  const checkHealth = useCallback(async () => {
    setHealthStatus("checking");
    try {
      const health = await ApiService.checkHealth();
      const nextStatus = health.status === "healthy" ? "healthy" : "error";
      setHealthStatus(nextStatus);
      if (nextStatus === "healthy") {
        errorToastShownRef.current = false;
      }
    } catch (error) {
      console.error("Health check failed:", error);
      setHealthStatus("error");
      if (!errorToastShownRef.current) {
        toast.error("Backend server is not available");
        errorToastShownRef.current = true;
      }
    }
  }, []);

  const loadRecentSessions = useCallback(async () => {
    setIsLoading(true);
    try {
      const sessions = await ApiService.listSessions(12);
      setRecentSessions(sessions);
    } catch (_) {
      toast.error("Failed to load recent sessions");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    loadRecentSessions();

    const poller = setInterval(() => {
      void checkHealth();
    }, 20000);

    return () => clearInterval(poller);
  }, [checkHealth, loadRecentSessions]);

  const handleStartAnalysis = () => {
    resetSession();
    navigate("/chat");
  };

  const handleOpenSession = (sessionId: string) => {
    resetSession();
    navigate(`/chat/${sessionId}`);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const connectionIndicator = useMemo(() => {
    if (isSocketConnected) {
      return {
        color: "bg-green-500",
        label: "Connected",
      };
    }

    if (healthStatus === "error") {
      return {
        color: "bg-red-500",
        label: "Offline",
      };
    }

    if (healthStatus === "checking") {
      return {
        color: "bg-yellow-500 animate-pulse",
        label: "Checking...",
      };
    }

    return {
      color: "bg-amber-400",
      label: "API Ready",
    };
  }, [healthStatus, isSocketConnected]);

  const getSessionTitle = (session: Session, index: number) => {
    if (session.metadata?.algorithm_name) {
      return session.metadata.algorithm_name;
    }

    if (session.metadata?.paradigm) {
      const paradigmName =
        typeof session.metadata.paradigm === "string"
          ? session.metadata.paradigm
          : session.metadata.paradigm.name;
      return `${paradigmName.replace(/_/g, " ")} analysis`;
    }

    return `Analysis ${index + 1}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-500/20 text-green-400 border-green-500/30";
      case "processing":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
      case "failed":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  const getSessionDescription = (session: Session) => {
    if (session.metadata?.latest_complexity) {
      const { worst_case, average_case, best_case } =
        session.metadata.latest_complexity;

      const parts = [
        worst_case ? `Worst ${worst_case}` : null,
        average_case ? `Avg ${average_case}` : null,
        best_case ? `Best ${best_case}` : null,
      ].filter(Boolean);

      if (parts.length > 0) {
        return parts.join(" • ");
      }
    }

    const text = session.user_input || "Sin descripción";
    return text.length > 110 ? `${text.slice(0, 107)}...` : text;
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-card shadow-lg shadow-cyan-500/5">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <Brain className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">
              Algorithm Complexity Analyzer
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${connectionIndicator.color}`}
            />
            <span className="text-sm text-muted-foreground">
              {connectionIndicator.label}
            </span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border-b border-primary/20 mb-6">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium text-primary">
              AI-Powered Multi-Agent Analysis
            </span>
          </div>

          <h2 className="text-5xl font-bold mb-12 text-balance mx-auto">
            Analyze Algorithm Complexity with Precision
          </h2>

          <p className="text-xl text-muted-foreground mb-8 text-balance leading-relaxed">
            Transform your pseudocode or natural language algorithm descriptions
            into comprehensive complexity analysis with <strong>O</strong>,{" "}
            <strong>Ω</strong>, and <strong>Θ</strong> notation, powered by
            advanced LLM agents.
          </p>

          <Button
            onClick={handleStartAnalysis}
            size="lg"
            className="group text-lg px-8 py-4 h-auto border border-primary/60 rounded-full font-semibold tracking-wide text-primary transition-all duration-200 hover:bg-primary/10 hover:-translate-y-0.5 hover:shadow-[0_0_25px_rgba(45,212,191,0.35)] focus-visible:ring-2 focus-visible:ring-primary/40 disabled:opacity-60"
            disabled={healthStatus === "error"}
          >
            <Code className="mr-2 h-5 w-5" />
            Start New Analysis
          </Button>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-16 max-w-5xl mx-auto">
          <Card className="p-6 bg-card border-border">
            <div className="p-3 rounded-lg bg-blue-500/10 w-fit mb-4">
              <Brain className="h-6 w-6 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Multi-Agent System</h3>
            <p className="text-muted-foreground text-sm">
              Six specialized AI agents work in sequence: Translator, Parser,
              Classifier, Analyzer, Validator, and Documenter.
            </p>
          </Card>

          <Card className="p-6 bg-card border-border">
            <div className="p-3 rounded-lg bg-purple-500/10 w-fit mb-4">
              <Sparkles className="h-6 w-6 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Human-in-the-Loop</h3>
            <p className="text-muted-foreground text-sm">
              Review and approve each agent's output before proceeding, ensuring
              accuracy and control over the analysis.
            </p>
          </Card>

          <Card className="p-6 bg-card border-border">
            <div className="p-3 rounded-lg bg-green-500/10 w-fit mb-4">
              <Code className="h-6 w-6 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Visual Diagrams</h3>
            <p className="text-muted-foreground text-sm">
              Automatic generation of recursion trees, flowcharts, and analysis
              visualizations using Mermaid.js.
            </p>
          </Card>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-semibold">Recent Analyses</h3>
            {recentSessions.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={loadRecentSessions}
                disabled={isLoading}
                className="border border-border/70 rounded-full px-4 py-1 text-xs font-semibold text-foreground hover:bg-border/20"
              >
                Refresh
              </Button>
            )}
          </div>

          {isLoading ? (
            <div className="text-center py-12 text-muted-foreground">
              Loading sessions...
            </div>
          ) : recentSessions.length === 0 ? (
            <Card className="p-12 text-center border-dashed">
              <p className="text-muted-foreground mb-4">
                No previous analyses found
              </p>
              <Button variant="outline" onClick={handleStartAnalysis}>
                Create Your First Analysis
              </Button>
            </Card>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {recentSessions.map((session, index) => (
                <Card
                  key={session.session_id}
                  className="p-4 h-full border-border bg-card/60 backdrop-blur-sm hover:shadow-lg hover:shadow-primary/10 transition-all cursor-pointer"
                  onClick={() => handleOpenSession(session.session_id)}
                >
                  <div className="flex items-start justify-between mb-3 gap-3">
                    <p className="text-sm font-semibold text-foreground leading-tight line-clamp-2">
                      {getSessionTitle(session, index)}
                    </p>
                    <span
                      className={`px-2 py-1 rounded-full text-[11px] uppercase tracking-wide border ${getStatusColor(session.status)}`}
                    >
                      {session.status}
                    </span>
                  </div>

                  <p className="text-xs text-muted-foreground mb-3">
                    {formatDate(session.created_at)}
                  </p>

                  <p className="text-xs text-muted-foreground line-clamp-4 leading-relaxed">
                    {getSessionDescription(session)}
                  </p>
                </Card>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
