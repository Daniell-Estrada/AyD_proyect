import { useEffect, useRef, useState } from "react";
import type { ChatWindowProps, HitlRequest } from "../../../lib/types";
import { Bot, User, Settings, AlertCircle } from "lucide-react";
import { Card } from "../ui/card";
import MarkdownRenderer from "../common/markdown.renderer";
import { Textarea } from "../ui/textarea";

interface HitlActionSectionProps {
  hitlRequest: HitlRequest;
  onHitlApprove: () => void;
  onHitlDeny: (feedback: string) => void;
  onHitlEdit: (editedOutput: any) => void;
  onHitlRecommend: (recommendation: string) => void;
}

function HitlActionSection({
  hitlRequest,
  onHitlApprove,
  onHitlDeny,
  onHitlEdit,
  onHitlRecommend,
}: HitlActionSectionProps) {
  const [mode, setMode] = useState<"idle" | "deny" | "edit" | "recommend">("idle");
  const [feedback, setFeedback] = useState("");
  const [recommendation, setRecommendation] = useState("");
  const [editedOutput, setEditedOutput] = useState<string>(() => {
    if (hitlRequest.output === null || hitlRequest.output === undefined) {
      return "";
    }

    return typeof hitlRequest.output !== "string"
      ? JSON.stringify(hitlRequest.output, null, 2)
      : String(hitlRequest.output);
  });
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    setMode("idle");
    setFeedback("");
    setRecommendation("");
    if (hitlRequest.output === null || hitlRequest.output === undefined) {
      setEditedOutput("");
    } else if (typeof hitlRequest.output === "string") {
      setEditedOutput(hitlRequest.output);
    } else {
      setEditedOutput(JSON.stringify(hitlRequest.output, null, 2));
    }
    setJsonError(null);
  }, [hitlRequest]);

  const handleEditChange = (value: string) => {
    setEditedOutput(value);
    if (typeof hitlRequest.output === "string") {
      return;
    }

    try {
      JSON.parse(value);
      setJsonError(null);
    } catch (error) {
      setJsonError("JSON inválido");
    }
  };

  const handleEditSubmit = () => {
    if (typeof hitlRequest.output === "string") {
      onHitlEdit(editedOutput);
      setMode("idle");
      return;
    }

    try {
      const parsed = JSON.parse(editedOutput);
      onHitlEdit(parsed);
      setMode("idle");
    } catch (error) {
      setJsonError("JSON inválido");
    }
  };

  return (
    <div className="mt-4 space-y-4">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        Selecciona una acción
      </p>

      <div className="flex flex-wrap gap-4 text-sm font-medium">
        <button
          type="button"
          onClick={onHitlApprove}
          className="text-green-400 hover:text-green-300"
        >
          Confirmar
        </button>
        <button
          type="button"
          onClick={() => setMode(mode === "deny" ? "idle" : "deny")}
          className="text-red-400 hover:text-red-300"
        >
          Rechazar
        </button>
        <button
          type="button"
          onClick={() => setMode(mode === "edit" ? "idle" : "edit")}
          className="text-blue-400 hover:text-blue-300"
        >
          Editar
        </button>
        <button
          type="button"
          onClick={() => setMode(mode === "recommend" ? "idle" : "recommend")}
          className="text-amber-400 hover:text-amber-300"
        >
          Recomendar
        </button>
      </div>

      {mode === "deny" && (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">Explica el motivo</p>
          <Textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Describe por qué rechazas este resultado"
            className="min-h-[100px] bg-background"
          />
          <button
            type="button"
            onClick={() => {
              const value = feedback.trim();
              if (!value) return;
              onHitlDeny(value);
              setMode("idle");
              setFeedback("");
            }}
            className="text-red-400 hover:text-red-300 text-sm font-semibold"
          >
            Enviar rechazo
          </button>
        </div>
      )}

      {mode === "edit" && (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            Ajusta la salida {typeof hitlRequest.output === "string" ? "como texto" : "en formato JSON"}
          </p>
          <Textarea
            value={editedOutput}
            onChange={(e) => handleEditChange(e.target.value)}
            className="font-mono text-xs min-h-[180px] bg-background whitespace-pre-wrap"
          />
          {typeof hitlRequest.output !== "string" && jsonError && (
            <p className="text-xs text-destructive">{jsonError}</p>
          )}
          <button
            type="button"
            onClick={handleEditSubmit}
            className="text-blue-400 hover:text-blue-300 text-sm font-semibold disabled:text-muted-foreground"
            disabled={typeof hitlRequest.output !== "string" && !!jsonError}
          >
            Aplicar edición
          </button>
        </div>
      )}

      {mode === "recommend" && (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            Comparte sugerencias o mejoras
          </p>
          <Textarea
            value={recommendation}
            onChange={(e) => setRecommendation(e.target.value)}
            placeholder="Ejemplo: revisa la descomposición del caso peor"
            className="min-h-[100px] bg-background"
          />
          <button
            type="button"
            onClick={() => {
              const suggestion = recommendation.trim();
              if (!suggestion) return;
              onHitlRecommend(suggestion);
              setRecommendation("");
              setMode("idle");
            }}
            className="text-amber-400 hover:text-amber-300 text-sm font-semibold"
          >
            Enviar recomendación
          </button>
        </div>
      )}
    </div>
  );
}

interface HitlContentProps {
  metadata?: Record<string, any>;
  isActive: boolean;
  pendingHitl: HitlRequest | null;
  onHitlApprove: () => void;
  onHitlDeny: (feedback: string) => void;
  onHitlEdit: (editedOutput: any) => void;
  onHitlRecommend: (recommendation: string) => void;
}

function HitlContent({
  metadata,
  isActive,
  pendingHitl,
  onHitlApprove,
  onHitlDeny,
  onHitlEdit,
  onHitlRecommend,
}: HitlContentProps) {
  if (!metadata) {
    return (
      <p className="text-sm text-muted-foreground">
        No hay información adicional para esta solicitud.
      </p>
    );
  }

  const reasoning = metadata?.reasoning;
  const output = metadata?.output;

  const renderReasoning = () => {
    if (!reasoning) return "Sin justificación";
    if (typeof reasoning === "string") return reasoning;
    try {
      return JSON.stringify(reasoning, null, 2);
    } catch (error) {
      return "No se pudo mostrar el razonamiento";
    }
  };

  const renderOutput = () => {
    if (output === undefined || output === null) {
      const preview = metadata?.output_preview;
      if (preview) {
        return preview;
      }
      return "No se recibió salida del agente.";
    }

    if (typeof output === "string") {
      return output;
    }

    try {
      return JSON.stringify(output, null, 2);
    } catch (error) {
      return "No se pudo mostrar la salida.";
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <p className="text-xs font-semibold text-muted-foreground mb-2">
          Razonamiento
        </p>
        <Card className="p-3 bg-background/40 border-border/50">
          <pre className="text-xs text-foreground whitespace-pre-wrap font-sans">
            {renderReasoning()}
          </pre>
        </Card>
      </div>

      <div>
        <p className="text-xs font-semibold text-muted-foreground mb-2">
          Salida propuesta
        </p>
        <Card className="p-3 bg-background/40 border-border/50 overflow-x-auto">
          <pre className="text-xs text-foreground whitespace-pre-wrap font-mono">
            {renderOutput()}
          </pre>
        </Card>
      </div>

      {isActive && pendingHitl && (
        <HitlActionSection
          hitlRequest={pendingHitl}
          onHitlApprove={onHitlApprove}
          onHitlDeny={onHitlDeny}
          onHitlEdit={onHitlEdit}
          onHitlRecommend={onHitlRecommend}
        />
      )}

      {!isActive && (
        <p className="text-xs text-muted-foreground">
          Esta solicitud ya fue atendida.
        </p>
      )}
    </div>
  );
}

export default function ChatWindow({
  messages,
  isAnalyzing,
  isLoadingSession,
  pendingHitl,
  onHitlApprove,
  onHitlDeny,
  onHitlEdit,
  onHitlRecommend,
}: ChatWindowProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getMessageIcon = (type: string) => {
    switch (type) {
      case "user":
        return <User className="h-5 w-5" />;
      case "agent":
        return <Bot className="h-5 w-5" />;
      case "hitl":
        return <AlertCircle className="h-5 w-5" />;
      case "system":
        return <Settings className="h-5 w-5" />;
      default:
        return <Bot className="h-5 w-5" />;
    }
  };

  const getMessageStyle = (type: string) => {
    switch (type) {
      case "user":
        return "bg-transparent text-primary ml-auto";
      case "agent":
        return "bg-transparent border-blue-500/40 text-foreground";
      case "hitl":
        return "bg-transparent border-amber-500/40 text-foreground";
      case "system":
        return "bg-transparent text-muted-foreground";
      default:
        return "bg-transparent text-foreground";
    }
  };

  const formatDetailValue = (value: unknown): string => {
    if (value === null || value === undefined) {
      return "";
    }

    if (typeof value === "string") {
      return value;
    }

    if (typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }

    try {
      return JSON.stringify(value, null, 2);
    } catch (error) {
      return String(value);
    }
  };

  const renderAgentDetails = (metadata?: Record<string, any>) => {
    if (!metadata) return null;

    const reasoningValue =
      metadata.reasoning ??
      metadata.analysis_reasoning ??
      metadata.explanation ??
      metadata.details?.reasoning;

    const outputValue =
      metadata.output ??
      metadata.output_preview ??
      metadata.result ??
      metadata.analysis ??
      metadata.final_output;

    const confidenceValue =
      typeof metadata.confidence === "number"
        ? metadata.confidence
        : typeof metadata.validation?.confidence === "number"
          ? metadata.validation.confidence
          : undefined;

    const normalizedConfidence =
      confidenceValue !== undefined
        ? confidenceValue > 1
          ? confidenceValue
          : confidenceValue * 100
        : undefined;

    const issuesValue = Array.isArray(metadata.issues)
      ? metadata.issues
      : Array.isArray(metadata.output?.issues)
        ? metadata.output.issues
        : undefined;

    const retryReason =
      metadata.retry_reason || metadata.analysis_retry_reason || metadata.retryReason;

    if (!reasoningValue && !outputValue) return null;

    return (
      <div className="mt-3 space-y-3">
        {reasoningValue && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground mb-2">
              Reasoning
            </p>
            <Card className="p-3 bg-background/40 border-border/50">
              <pre className="text-xs text-foreground whitespace-pre-wrap font-sans">
                {formatDetailValue(reasoningValue)}
              </pre>
            </Card>
          </div>
        )}

        {outputValue && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground mb-2">
              Proposed Output
            </p>
            <Card className="p-3 bg-background/40 border-border/50 overflow-x-auto">
              <pre className="text-xs text-foreground whitespace-pre-wrap">
                {formatDetailValue(outputValue)}
              </pre>
            </Card>
          </div>
        )}

        {issuesValue && issuesValue.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground mb-1">
              Notas
            </p>
            <ul className="list-disc pl-5 space-y-1 text-xs text-muted-foreground">
              {issuesValue.map((issue, index) => (
                <li key={index}>{formatDetailValue(issue)}</li>
              ))}
            </ul>
          </div>
        )}

        {retryReason && (
          <div className="flex items-center gap-2 text-xs text-amber-500">
            <span className="px-2 py-1 rounded-full border border-amber-500/40 bg-amber-500/10">
              Retry reason: {formatDetailValue(retryReason)}
            </span>
          </div>
        )}

        {normalizedConfidence !== undefined && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Confidence:</span>
            <span className="font-medium text-foreground">
              {normalizedConfidence.toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    );
  };

  const resolveMetric = (
    metadata: Record<string, any> | undefined,
    keys: string[],
  ): number | undefined => {
    if (!metadata) return undefined;

    for (const key of keys) {
      const value = metadata[key];
      if (typeof value === "number") {
        return value;
      }
    }

    if (metadata.metadata) {
      for (const key of keys) {
        const nested = metadata.metadata[key];
        if (typeof nested === "number") {
          return nested;
        }
      }
    }

    if (metadata.metrics) {
      for (const key of keys) {
        const nested = metadata.metrics[key];
        if (typeof nested === "number") {
          return nested;
        }
      }
    }

    return undefined;
  };

  if (isLoadingSession) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center text-muted-foreground">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
          <p>Loading session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 pb-32 space-y-4">
      {messages.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full text-center p-8">
          <Bot className="h-16 w-16 text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2 text-foreground">
            Ready to Analyze
          </h3>
          <p className="text-muted-foreground max-w-md">
            Enter your algorithm in natural language or pseudocode below to
            begin complexity analysis
          </p>
        </div>
      ) : (
        <>
          {messages.map((message) => {
            const isActiveHitl =
              message.type === "hitl" &&
              pendingHitl &&
              message.metadata?.stage === pendingHitl.stage;

            return (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === "user" ? "justify-end" : "justify-start"}`}
              >
              {message.type !== "user" && (
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                    message.type === "agent"
                      ? "bg-blue-500/20 text-blue-400"
                      : message.type === "hitl"
                        ? "bg-amber-500/20 text-amber-400"
                        : "bg-muted text-muted-foreground"
                  }`}
                >
                  {getMessageIcon(message.type)}
                </div>
              )}

                <div
                  className={`flex flex-col max-w-[75%] ${message.type === "user" ? "items-end" : "items-start"}`}
                >
                {message.agent_name && (
                  <span className="text-xs text-muted-foreground mb-1 px-1">
                    {message.agent_name}
                  </span>
                )}

                <Card className={`px-0 py-0 border-0 shadow-none ${getMessageStyle(message.type)}`}>
                  {message.type === "hitl" ? (
                    <HitlContent
                      metadata={message.metadata}
                      isActive={Boolean(isActiveHitl)}
                      pendingHitl={pendingHitl}
                      onHitlApprove={onHitlApprove}
                      onHitlDeny={onHitlDeny}
                      onHitlEdit={onHitlEdit}
                      onHitlRecommend={onHitlRecommend}
                    />
                  ) : (
                    <>
                      <MarkdownRenderer content={message.content} />
                      {message.type === "agent" &&
                        renderAgentDetails(message.metadata)}
                    </>
                  )}

                  {(() => {
                    const costValue = resolveMetric(message.metadata, [
                      "cost_usd",
                      "total_cost_usd",
                    ]);
                    const tokenValue = resolveMetric(message.metadata, [
                      "tokens",
                      "total_tokens",
                    ]);
                    const durationValue = resolveMetric(message.metadata, [
                      "duration_ms",
                      "total_duration_ms",
                    ]);

                    if (
                      costValue === undefined &&
                      tokenValue === undefined &&
                      durationValue === undefined
                    ) {
                      return null;
                    }

                    return (
                      <div className="flex gap-3 mt-2 pt-2 border-t border-border/50 text-xs text-muted-foreground flex-wrap">
                        {costValue !== undefined && (
                          <span>${costValue.toFixed(4)}</span>
                        )}
                        {tokenValue !== undefined && (
                          <span>{Math.round(tokenValue).toLocaleString()} tokens</span>
                        )}
                        {durationValue !== undefined && (
                          <span>
                            {(durationValue / 1000).toFixed(1)}s
                          </span>
                        )}
                      </div>
                    );
                  })()}
                </Card>

                <span className="text-xs text-muted-foreground mt-1 px-1">
                  {formatTime(message.timestamp)}
                </span>
                </div>

              {message.type === "user" && (
                <div className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-primary text-primary-foreground">
                  <User className="h-5 w-5" />
                </div>
              )}
              </div>
            );
          })}

          {isAnalyzing && (
            <div className="flex gap-3">
              <div className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-blue-500/20 text-blue-400">
                <Bot className="h-5 w-5" />
              </div>
              <Card className="px-4 py-3 bg-blue-500/10 border-blue-500/20">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                  </div>
                  <span className="text-sm text-foreground">Processing...</span>
                </div>
              </Card>
            </div>
          )}

          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
}
