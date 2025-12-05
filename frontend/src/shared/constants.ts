import {
  BookA,
  TreeDeciduous,
  Shapes,
  ChartNetwork,
  TicketCheck,
  BookOpenText,
} from "lucide-react";

export const SOCKET_EVENTS = {
  START_ANALYSIS: "start_analysis",
  HITL_RESPOND: "hitl_respond",

  CONNECTION_ESTABLISHED: "connection_established",
  SESSION_CREATED: "session_created",
  AGENT_UPDATE: "agent_update",
  HITL_REQUEST: "hitl_request",
  ANALYSIS_COMPLETE: "analysis_complete",
  ANALYSIS_FAILED: "analysis_failed",
  ERROR: "error",
} as const;

export const AGENT_STAGES = [
  {
    key: "translation",
    name: "Traductor",
    description: "Convierte lenguaje natural a pseudocódigo",
    icon: BookA,
  },
  {
    key: "parsing",
    name: "Parser",
    description: "Genera el Abstract Syntax Tree (AST)",
    icon: TreeDeciduous,
  },
  {
    key: "classification",
    name: "Clasificador",
    description: "Identifica el paradigma algorítmico",
    icon: Shapes,
  },
  {
    key: "analysis",
    name: "Analizador",
    description: "Calcula complejidades temporales y espaciales",
    icon: ChartNetwork,
  },
  {
    key: "validation",
    name: "Validador",
    description: "Valida resultados matemáticamente",
    icon: TicketCheck,
  },
  {
    key: "documentation",
    name: "Documentador",
    description: "Genera documentación y diagramas",
    icon: BookOpenText,
  },
] as const;

export const UI_CONFIG = {
  MAX_MESSAGE_LENGTH: 10000,
  TOAST_DURATION: 4000,
  SCROLL_BEHAVIOR: "smooth" as ScrollBehavior,
  DEBOUNCE_DELAY: 300,
} as const;

export const STATUS_COLORS = {
  pending: {
    bg: "bg-yellow-500/20",
    text: "text-yellow-400",
    border: "border-yellow-500/30",
  },
  processing: {
    bg: "bg-blue-500/20",
    text: "text-blue-400",
    border: "border-blue-500/30",
  },
  completed: {
    bg: "bg-green-500/20",
    text: "text-green-400",
    border: "border-green-500/30",
  },
  failed: {
    bg: "bg-red-500/20",
    text: "text-red-400",
    border: "border-red-500/30",
  },
  started: {
    bg: "bg-blue-500/20",
    text: "text-blue-400",
    border: "border-blue-500/30",
  },
} as const;

export const COMPLEXITY_COLORS = {
  worst_case: {
    bg: "bg-red-500/10",
    text: "text-red-400",
    border: "border-red-500/30",
    label: "Peor Caso",
    notation: "O",
  },
  best_case: {
    bg: "bg-green-500/10",
    text: "text-green-400",
    border: "border-green-500/30",
    label: "Mejor Caso",
    notation: "Ω",
  },
  average_case: {
    bg: "bg-blue-500/10",
    text: "text-blue-400",
    border: "border-blue-500/30",
    label: "Caso Promedio",
    notation: "Θ",
  },
  tight_bounds: {
    bg: "bg-purple-500/10",
    text: "text-purple-400",
    border: "border-purple-500/30",
    label: "Cota Ajustada",
    notation: "Θ",
  },
} as const;

export const ERROR_MESSAGES = {
  CONNECTION_FAILED:
    "No se pudo conectar con el servidor. Por favor, intenta de nuevo.",
  SESSION_NOT_FOUND: "Sesión no encontrada.",
  ANALYSIS_FAILED:
    "Error al analizar el algoritmo. Revisa la entrada e intenta nuevamente.",
  HITL_REQUIRED: "Debes aprobar, rechazar o editar antes de continuar.",
  FEEDBACK_REQUIRED: "Debes proporcionar retroalimentación al rechazar.",
  INVALID_INPUT: "La entrada proporcionada no es válida.",
  NETWORK_ERROR: "Error de red. Verifica tu conexión.",
} as const;

export const SUCCESS_MESSAGES = {
  CONNECTED: "Conectado al servidor",
  ANALYSIS_COMPLETE: "Análisis completado exitosamente",
  HITL_APPROVED: "Aprobado exitosamente",
  HITL_DENIED: "Rechazado. El agente reintentará.",
  HITL_EDITED: "Edición enviada exitosamente",
} as const;
