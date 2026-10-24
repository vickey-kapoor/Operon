// GET /api/runs: every benchmark run with its summary and config, oldest first.
import { db, handle, STAMP, tablesExist } from "../server/db.js";

export default function handler(req, res) {
  return handle(res, async () => {
    if (!(await tablesExist())) return { generated_at: null, runs: [] };
    const { rows } = await db().query(
      `SELECT key, suite, mode, ${STAMP} AS started_at, judge_model, summary, config
         FROM bench_runs ORDER BY started_at, key`,
    );
    const latest = await db().query("SELECT max(imported_at) AS at FROM bench_runs");
    return { generated_at: latest.rows[0].at, runs: rows };
  });
}
