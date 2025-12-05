export const environment = {
  BASE_URL: import.meta.env.VITE_BASE_URL || "http://localhost:5000",
  WS_URL: import.meta.env.VITE_WS_URL || "http://localhost:5000",
  TIMEOUT: Number(import.meta.env.VITE_TIMEOUT) || 30000,
};
