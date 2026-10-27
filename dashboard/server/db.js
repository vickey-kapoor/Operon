// Shared Postgres access for the API routes. Vercel's Neon integration sets
// DATABASE_URL (POSTGRES_URL on older setups); tests/eval/dashboard.py writes the
// tables. One small pool per function instance is plenty for a dashboard.
import pg from "pg";

let pool;

export function db() {
  const connectionString = process.env.DATABASE_URL || process.env.POSTGRES_URL;
  if (!connectionString) throw new Error("DATABASE_URL is not set");
  pool ??= new pg.Pool({ connectionString, max: 2, idleTimeoutMillis: 10_000 });
  return pool;
}

// live_bench's own stamp format (20261024T012945Z), which the page parses.
export const STAMP = `to_char(started_at AT TIME ZONE 'UTC', 'YYYYMMDD"T"HH24MISS"Z"')`;

// Before the first import the tables don't exist yet; the page then shows no runs.
export async function tablesExist() {
  const { rows } = await db().query("SELECT to_regclass('bench_runs') IS NOT NULL AS ok");
  return rows[0].ok;
}

export function sendJson(res, status, body) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  // Behind Vercel Authentication, so only the viewer's browser may cache it.
  res.setHeader("Cache-Control", status === 200 ? "private, max-age=60" : "no-store");
  res.end(JSON.stringify(body));
}

export async function handle(res, fn) {
  try {
    sendJson(res, 200, await fn());
  } catch (err) {
    if (err.status) return sendJson(res, err.status, { error: err.message });
    console.error(err);
    sendJson(res, 500, { error: "Could not read the benchmark database" });
  }
}
