/**
 * Connection settings for the Operon API.
 *
 * When the API runs with API_KEYS set, put one key in `ui/.env.local` as
 * VITE_OPERON_API_KEY. HTTP calls send it as `X-API-Key`; the WebSocket offers
 * it as the `x-api-key` subprotocol, because browsers can't set headers on a
 * WebSocket and a `?api_key=` query string would land in server access logs.
 */
export const API = "http://127.0.0.1:8080";

const API_KEY = import.meta.env.VITE_OPERON_API_KEY ?? "";

export function apiHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  return headers;
}

export function wsProtocols(): string[] | undefined {
  return API_KEY ? ["x-api-key", API_KEY] : undefined;
}
