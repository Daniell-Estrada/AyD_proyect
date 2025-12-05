import { useState } from "react";
import type { AnalysisResult, ParadigmDescriptor } from "../../../lib/types";
import { Card } from "../ui/card";
import MermaidDiagram from "./mermaid.diagram";
import {
  ChevronDown,
  ChevronUp,
  TrendingUp,
  CheckCircle2,
  Clock,
  DollarSign,
  Zap,
} from "lucide-react";

interface ResultsPanelProps {
  result: AnalysisResult;
}

export default function ResultsPanel({ result }: ResultsPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(["complexity", "diagrams"]),
  );

  const isParadigmDescriptor = (
    value: AnalysisResult["paradigm"],
  ): value is ParadigmDescriptor => typeof value === "object" && value !== null;

  const paradigmDescriptor = isParadigmDescriptor(result.paradigm)
    ? result.paradigm
    : null;

  const paradigmName =
    paradigmDescriptor?.name ?? String(result.paradigm ?? "");
  const paradigmLabel = paradigmName
    ? paradigmName.replace(/_/g, " ").toUpperCase()
    : "";
  const paradigmConfidence = paradigmDescriptor?.confidence;
  const paradigmReasoning =
    paradigmDescriptor?.reasoning ??
    (typeof result.metadata?.paradigm_reasoning === "string"
      ? result.metadata?.paradigm_reasoning
      : undefined);

  const metricSource = result.metrics ?? result.metadata ?? {};

  const totalCost =
    typeof metricSource?.total_cost_usd === "number"
      ? metricSource.total_cost_usd
      : typeof metricSource?.cost_usd === "number"
        ? metricSource.cost_usd
        : null;

  const totalTokens =
    typeof metricSource?.total_tokens === "number"
      ? metricSource.total_tokens
      : typeof metricSource?.tokens === "number"
        ? metricSource.tokens
        : null;

  const totalDurationMs =
    typeof metricSource?.total_duration_ms === "number"
      ? metricSource.total_duration_ms
      : typeof metricSource?.duration_ms === "number"
        ? metricSource.duration_ms
        : null;

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const isExpanded = (section: string) => expandedSections.has(section);

  return (
    <div className="p-6 space-y-6 bg-background">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">
          {result.algorithm_name}
        </h2>
        <div className="flex items-center gap-2">
          {paradigmLabel && (
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30 flex items-center gap-1">
              <span>{paradigmLabel}</span>
              {typeof paradigmConfidence === "number" && (
                <span className="text-[10px] text-purple-200/80">
                  {Math.round(paradigmConfidence * 100)}%
                </span>
              )}
            </span>
          )}
          {result.validation?.valid && (
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/30 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Validated
            </span>
          )}
        </div>

        {paradigmReasoning && (
          <p className="text-sm text-muted-foreground mt-3">
            {paradigmReasoning}
          </p>
        )}
      </div>

      <Card className="border-border bg-card">
        <button
          onClick={() => toggleSection("complexity")}
          className="w-full px-4 py-3 flex items-center justify-between hover:bg-accent/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            <h3 className="font-semibold text-foreground">Complexity</h3>
          </div>
          {isExpanded("complexity") ? (
            <ChevronUp className="h-5 w-5 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-5 w-5 text-muted-foreground" />
          )}
        </button>

        {isExpanded("complexity") && (
          <div className="px-4 pb-4 space-y-3">
            {result.complexity.worst_case && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Worst Case:
                </span>
                <code className="text-sm font-mono font-semibold text-red-400">
                  {result.complexity.worst_case}
                </code>
              </div>
            )}
            {result.complexity.best_case && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Best Case:
                </span>
                <code className="text-sm font-mono font-semibold text-green-400">
                  {result.complexity.best_case}
                </code>
              </div>
            )}
            {result.complexity.average_case && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Average Case:
                </span>
                <code className="text-sm font-mono font-semibold text-blue-400">
                  {result.complexity.average_case}
                </code>
              </div>
            )}
          </div>
        )}
      </Card>

      {result.analysis_steps && result.analysis_steps.length > 0 && (
        <Card className="border-border bg-card">
          <button
            onClick={() => toggleSection("steps")}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-accent/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              <h3 className="font-semibold text-foreground">Analysis Steps</h3>
            </div>
            {isExpanded("steps") ? (
              <ChevronUp className="h-5 w-5 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-5 w-5 text-muted-foreground" />
            )}
          </button>

          {isExpanded("steps") && (
            <div className="px-6 pb-6">
              <div className="relative grid gap-8">
                {result.analysis_steps.map((step, index) => {
                  const stepLabel = step.step || index + 1;
                  const hasTokens = typeof step.tokens === "number";
                  const hasCost = typeof step.cost_usd === "number";

                  const isLast = index === result.analysis_steps.length - 1;

                  return (
                    <div
                      key={`${step.technique}-${index}`}
                      className={`relative pl-10 border-l ${isLast ? "border-transparent" : "border-border"}`}
                    >
                      <span className="absolute -left-2 top-1 w-4 h-4 rounded-full border-2 border-primary bg-background" />
                      <div className="rounded-lg border border-border bg-background/70 p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-3 flex-wrap">
                          <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                            <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary font-semibold">
                              #{stepLabel}
                            </span>
                            <span>{String(step.technique || "Unknown")}</span>
                          </div>
                          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                            {hasTokens && (
                              <span>
                                Tokens: {Math.round(step.tokens!).toLocaleString()}
                              </span>
                            )}
                            {hasCost && (
                              <span>
                                Cost: ${step.cost_usd!.toFixed(4)}
                              </span>
                            )}
                          </div>
                        </div>
                        {step.description && (
                          <p className="text-sm text-foreground mt-3 leading-relaxed">
                            {String(step.description)}
                          </p>
                        )}
                        {step.result && (
                          <div className="mt-3 bg-muted/40 border border-border rounded-md p-3 overflow-x-auto">
                            <code className="text-xs font-mono text-primary whitespace-pre-wrap break-words block">
                              {typeof step.result === "string"
                                ? step.result
                                : JSON.stringify(step.result, null, 2)}
                            </code>
                          </div>
                        )}
                        {step.explanation && (
                          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                            {String(step.explanation)}
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Card>
      )}

      {result.diagrams && Object.keys(result.diagrams).length > 0 && (
        <Card className="border-border bg-card">
          <button
            onClick={() => toggleSection("diagrams")}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-accent/50 transition-colors"
          >
            <h3 className="font-semibold text-foreground">Diagrams</h3>
            {isExpanded("diagrams") ? (
              <ChevronUp className="h-5 w-5 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-5 w-5 text-muted-foreground" />
            )}
          </button>

          {isExpanded("diagrams") && (
            <div className="px-4 pb-4 space-y-4">
              {Object.entries(result.diagrams)
                .filter(
                  ([, code]) =>
                    typeof code === "string" && code.trim().length > 0,
                )
                .map(([type, code]) => (
                  <div key={type}>
                    <h4 className="text-sm font-medium text-foreground mb-2 capitalize">
                      {type.replace(/_/g, " ")}
                    </h4>
                    <MermaidDiagram chart={code} />
                  </div>
                ))}
            </div>
          )}
        </Card>
      )}

      {result.pseudocode && (
        <Card className="border-border bg-card">
          <button
            onClick={() => toggleSection("pseudocode")}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-accent/50 transition-colors"
          >
            <h3 className="font-semibold text-foreground">Pseudocode</h3>
            {isExpanded("pseudocode") ? (
              <ChevronUp className="h-5 w-5 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-5 w-5 text-muted-foreground" />
            )}
          </button>

          {isExpanded("pseudocode") && (
            <div className="px-4 pb-4">
              <Card className="p-4 bg-background border-border overflow-x-auto">
                <pre className="text-xs font-mono text-foreground whitespace-pre-wrap">
                  {result.pseudocode}
                </pre>
              </Card>
            </div>
          )}
        </Card>
      )}

      <Card className="p-5 bg-gradient-to-br from-background/80 to-muted/60 border-border">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-center">
          <div>
            <div className="flex items-center justify-center gap-2 text-muted-foreground text-xs uppercase tracking-wide mb-1">
              <DollarSign className="h-4 w-4" />
              Cost
            </div>
            <p className="text-lg font-semibold text-foreground">
              {typeof totalCost === "number" ? `$${totalCost.toFixed(4)}` : "—"}
            </p>
          </div>
          <div>
            <div className="flex items-center justify-center gap-2 text-muted-foreground text-xs uppercase tracking-wide mb-1">
              <Zap className="h-4 w-4" />
              Tokens
            </div>
            <p className="text-lg font-semibold text-foreground">
              {typeof totalTokens === "number"
                ? Math.round(totalTokens).toLocaleString()
                : "—"}
            </p>
          </div>
          <div>
            <div className="flex items-center justify-center gap-2 text-muted-foreground text-xs uppercase tracking-wide mb-1">
              <Clock className="h-4 w-4" />
              Duration
            </div>
            <p className="text-lg font-semibold text-foreground">
              {typeof totalDurationMs === "number"
                ? `${(totalDurationMs / 1000).toFixed(1)}s`
                : "—"}
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
