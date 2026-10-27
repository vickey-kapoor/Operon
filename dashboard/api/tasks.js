// GET /api/tasks: every task result of every run, slimmed to what the charts use
// (task history, pass rate by category, wasted spend, priciest tasks).
import { db, handle, tablesExist } from "../server/db.js";

export default function handler(req, res) {
  return handle(res, async () => {
    if (!(await tablesExist())) return { tasks: [] };
    const { rows } = await db().query(
      `SELECT run_key, task, category, outcome, steps, seconds, cost_usd, judge_cost_usd,
              result->>'stop_reason' AS stop_reason
         FROM bench_task_results ORDER BY run_key, position`,
    );
    return { tasks: rows };
  });
}
