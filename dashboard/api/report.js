// GET /api/report?key=<run key>: one run with its per-task results, in run order.
import { db, handle, STAMP } from "../server/db.js";

export default function handler(req, res) {
  return handle(res, async () => {
    const key = new URL(req.url, "http://localhost").searchParams.get("key");
    if (!key) throw Object.assign(new Error("key is required"), { status: 400 });
    const run = await db().query(
      `SELECT key, suite, mode, ${STAMP} AS started_at, judge_model, summary, config
         FROM bench_runs WHERE key = $1`,
      [key],
    );
    if (!run.rows.length) throw Object.assign(new Error("No such run"), { status: 404 });
    const results = await db().query(
      "SELECT result FROM bench_task_results WHERE run_key = $1 ORDER BY position",
      [key],
    );
    return { ...run.rows[0], results: results.rows.map((row) => row.result) };
  });
}
