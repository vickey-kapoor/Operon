// GET /api/runs: every benchmark run with its summary and config, oldest first.
import { db, handle, STAMP, tablesExist } from "../server/db.js";

export default function handler(req, res) {
  return handle(res, async () => {
    if (!(await tablesExist())) return { generated_at: null, budget_usd: null, runs: [] };
    const { rows } = await db().query(
      `SELECT key, suite, mode, ${STAMP} AS started_at, judge_model, summary, config
         FROM bench_runs ORDER BY started_at, key`,
    );
    const latest = await db().query("SELECT max(imported_at) AS at FROM bench_runs");
    // Optional monthly budget for the Cost tab's projection, set in Vercel's env.
    const budget = Number(process.env.MONTHLY_BUDGET_USD) || null;
    return { generated_at: latest.rows[0].at, budget_usd: budget, runs: rows };
  });
}
