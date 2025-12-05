import { useState } from "react";
import type { HitlModalProps } from "../../../lib/types";
import { Button } from "../ui/button";
import { Card } from "../ui/card";
import { Textarea } from "../ui/textarea";
import { X, CheckCircle, XCircle, Edit } from "lucide-react";

export default function HitlModal({
  hitlRequest,
  onApprove,
  onDeny,
  onEdit,
}: HitlModalProps) {
  const [mode, setMode] = useState<"view" | "deny" | "edit">("view");
  const [feedback, setFeedback] = useState("");
  const [editedOutput, setEditedOutput] = useState(
    JSON.stringify(hitlRequest.output, null, 2),
  );
  const [jsonError, setJsonError] = useState("");

  const renderReasoning = () => {
    const reasoning = hitlRequest.reasoning;
    if (!reasoning) return "No reasoning provided";
    if (typeof reasoning === "string") return reasoning;
    return JSON.stringify(reasoning, null, 2);
  };

  const handleDenySubmit = () => {
    if (!feedback.trim()) {
      alert("Please provide feedback for denial");
      return;
    }
    onDeny(feedback);
  };

  const handleEditSubmit = () => {
    try {
      const parsed = JSON.parse(editedOutput);
      setJsonError("");
      onEdit(parsed);
    } catch (error) {
      setJsonError("Invalid JSON format");
    }
  };

  const handleEditChange = (value: string) => {
    setEditedOutput(value);
    try {
      JSON.parse(value);
      setJsonError("");
    } catch (error) {
      setJsonError("Invalid JSON");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
      <Card className="w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col border-border bg-card">
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-foreground">
              Human Approval Required
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {hitlRequest.agent_name} • {hitlRequest.stage}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setMode("view")}
            className="rounded-full"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div>
            <h3 className="text-sm font-semibold text-foreground mb-2">
              Reasoning:
            </h3>
            <Card className="p-4 bg-muted/50 border-border">
              <pre className="text-sm text-foreground whitespace-pre-wrap font-sans">
                {renderReasoning()}
              </pre>
            </Card>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-foreground mb-2">
              Output:
            </h3>
            {mode === "edit" ? (
              <div className="space-y-2">
                <Textarea
                  value={editedOutput}
                  onChange={(e) => handleEditChange(e.target.value)}
                  className="font-mono text-xs min-h-[200px] bg-background"
                />
                {jsonError && (
                  <p className="text-xs text-destructive">{jsonError}</p>
                )}
              </div>
            ) : (
              <Card className="p-4 bg-background border-border overflow-x-auto">
                <pre className="text-xs text-foreground">
                  {JSON.stringify(hitlRequest.output, null, 2)}
                </pre>
              </Card>
            )}
          </div>

          {mode === "deny" && (
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">
                Denial Reason:
              </h3>
              <Textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="Explain why you're denying this output..."
                className="min-h-[100px] bg-background"
              />
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-border flex items-center justify-end gap-3">
          {mode === "view" && (
            <>
              <Button
                variant="outline"
                onClick={() => setMode("deny")}
                className="gap-2"
              >
                <XCircle className="h-4 w-4" />
                Deny
              </Button>
              <Button
                variant="outline"
                onClick={() => setMode("edit")}
                className="gap-2"
              >
                <Edit className="h-4 w-4" />
                Edit
              </Button>
              <Button onClick={onApprove} className="gap-2">
                <CheckCircle className="h-4 w-4" />
                Approve
              </Button>
            </>
          )}

          {mode === "deny" && (
            <>
              <Button variant="outline" onClick={() => setMode("view")}>
                Cancel
              </Button>
              <Button
                onClick={handleDenySubmit}
                disabled={!feedback.trim()}
                variant="destructive"
                className="gap-2"
              >
                <XCircle className="h-4 w-4" />
                Confirm Denial
              </Button>
            </>
          )}

          {mode === "edit" && (
            <>
              <Button variant="outline" onClick={() => setMode("view")}>
                Cancel
              </Button>
              <Button
                onClick={handleEditSubmit}
                disabled={!!jsonError}
                className="gap-2"
              >
                <CheckCircle className="h-4 w-4" />
                Apply & Continue
              </Button>
            </>
          )}
        </div>
      </Card>
    </div>
  );
}
