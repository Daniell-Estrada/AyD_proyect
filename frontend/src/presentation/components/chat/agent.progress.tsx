import { useStore } from "../../../lib/store";
import {
  CheckCircle,
  Circle,
  Loader2,
  XCircle,
  PauseCircle,
} from "lucide-react";
import { AGENT_STAGES } from "../../../shared/constants";

interface AgentProgressProps {
  hasActiveSession?: boolean;
}

export default function AgentProgress({
  hasActiveSession = false,
}: AgentProgressProps) {
  const { agentEvents } = useStore();

  const activeStageKey = [...agentEvents]
    .reverse()
    .find((event) => event.status === "started")?.stage;

  const getAgentStatus = (agentStageKey: string) => {
    const events = agentEvents.filter((e) => e.stage === agentStageKey);
    if (events.length === 0) {
      return activeStageKey === agentStageKey ? "started" : "pending";
    }

    const latestEvent = events[events.length - 1];

    const hasFailed = events.some((event) => event.status === "failed");
    if (hasFailed) return "failed";

    const hasCompleted = events.some((event) =>
      ["completed", "resolved"].includes(event.status),
    );
    if (hasCompleted) {
      if (
        latestEvent?.status === "completed" &&
        latestEvent.payload &&
        "requires_review" in latestEvent.payload &&
        latestEvent.payload.requires_review
      ) {
        return "pending";
      }
      return "completed";
    }

    return activeStageKey === agentStageKey ? "started" : "pending";
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "started":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-400" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-400" />;
      default:
        return <Circle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "started":
        return "text-blue-400";
      case "completed":
        return "text-green-400";
      case "failed":
        return "text-red-400";
      default:
        return "text-muted-foreground";
    }
  };

  return (
    <div className="bg-card/95 backdrop-blur py-3">
      <div className="container mx-auto space-y-2">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          {AGENT_STAGES.map((stage, index) => {
            const status = getAgentStatus(stage.key);
            const StageIcon = stage.icon;

            return (
              <div key={stage.name} className="flex items-center gap-2 flex-1">
                <div className="flex items-center gap-2">
                  {getStatusIcon(status)}
                  <StageIcon className="h-4 w-4 text-muted-foreground" />
                  <span
                    className={`text-sm font-medium ${getStatusColor(status)}`}
                  >
                    {stage.name}
                  </span>
                </div>

                {index < AGENT_STAGES.length - 1 && (
                  <div
                    className={`flex-1 h-0.5 ${status === "completed" ? "bg-green-400" : "bg-border"}`}
                  />
                )}
              </div>
            );
          })}
        </div>

        {!hasActiveSession && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <PauseCircle className="h-3.5 w-3.5" />
            Inicia un análisis para seguir el progreso de los agentes.
          </div>
        )}
      </div>
    </div>
  );
}
