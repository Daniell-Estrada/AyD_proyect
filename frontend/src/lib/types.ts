export type SessionStatus = "pending" | "processing" | "completed" | "failed";

export interface Session {
  session_id: string;
  user_input: string;
  status: SessionStatus;
  current_stage: string;
  created_at: string;
  updated_at: string;
  total_cost_usd: number;
  total_tokens: number;
  total_duration_ms: number;
  hitl_approvals: HitlApproval[];
  metadata?: SessionMetadata;
}

export interface SessionMetadata extends Record<string, any> {
  algorithm_name?: string;
  paradigm?: string | ParadigmDescriptor;
  latest_complexity?: Partial<ComplexitySummary>;
  last_completed_at?: string;
}

export interface HitlApproval {
  stage: string;
  action: HitlAction;
  feedback?: string;
  timestamp: string;
}

export interface ComplexitySummary {
  worst_case: string | null;
  best_case: string | null;
  average_case: string | null;
  tight_bounds: string | null;
}

export interface AnalysisStep {
  step: number;
  technique: string;
  description: string;
  result: string;
  cost_usd?: number;
  tokens?: number;
}

export interface AnalysisResult {
  session_id: string;
  algorithm_name: string;
  pseudocode: string;
  ast: Record<string, any>;
  paradigm: string | ParadigmDescriptor;
  complexity: ComplexitySummary;
  analysis_steps: AnalysisStep[];
  diagrams: Record<string, string>;
  validation: ValidationResult;
  metadata: {
    total_cost_usd: number;
    total_tokens: number;
    total_duration_ms: number;
  };
  created_at: string;
}

export interface ParadigmDescriptor {
  name: string;
  confidence?: number;
  reasoning?: string;
}

export interface ValidationResult {
  valid: boolean;
  method: string;
  confidence: number;
  errors?: string[];
}

export type AgentEventStatus =
  | "started"
  | "completed"
  | "failed"
  | "pending"
  | "resolved";

export interface AgentEvent {
  session_id: string;
  stage: string;
  agent_name: string;
  status: AgentEventStatus;
  payload: {
    message?: string;
    cost_usd?: number;
    tokens?: number;
    duration_ms?: number;
    [key: string]: any;
  };
  timestamp?: string;
}

export type HitlAction = "approve" | "deny" | "edit" | "recommend";

export interface HitlRequest {
  session_id: string;
  stage: string;
  agent_name: string;
  output: any;
  reasoning: string;
}

export interface HitlResponse {
  session_id: string;
  action: HitlAction;
  feedback?: string;
  edited_output?: any;
  stage?: string;
}

export interface StartAnalysisPayload {
  user_input: string;
}

export interface SessionCreatedPayload {
  session_id: string;
}

export interface AnalysisCompletePayload {
  session_id: string;
  result: Partial<AnalysisResult>;
}

export interface AnalysisFailedPayload {
  session_id: string;
  errors: string[];
}

export interface ErrorPayload {
  session_id?: string;
  message: string;
}

export interface ChatInputProps {
  onSubmit: (userInput: string) => void;
  disabled?: boolean;
}

export interface ChatWindowProps {
  messages: ChatMessage[];
  isAnalyzing: boolean;
  isLoadingSession: boolean;
  pendingHitl: HitlRequest | null;
  onHitlApprove: () => void;
  onHitlDeny: (feedback: string) => void;
  onHitlEdit: (editedOutput: any) => void;
  onHitlRecommend: (recommendation: string) => void;
}

export interface ChatMessage {
  id: string;
  type: "user" | "agent" | "system" | "hitl";
  content: string;
  timestamp: Date;
  agent_name?: string;
  metadata?: Record<string, any>;
}

export interface DiagramData {
  type: string;
  mermaid_code: string;
}
