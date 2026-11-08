// GET /api/tasks: every task result of every run, slimmed to what the charts use
// (task history and its likely cause, pass rate by category, wasted spend, priciest tasks).
import { db, handle, tablesExist } from "../server/db.js";

export default function handler(req, res) {
  return handle(res, async () => {
    if (!(await tablesExist())) return { tasks: [] };
    const { rows } = await db().query(
      `SELECT run_key, task, category, outcome, steps, seconds, cost_usd, judge_cost_usd,
              result->>'stop_reason' AS stop_reason,
              -- Only misses need a reason (the "likely cause" column).
              CASE WHEN outcome <> 'success' THEN result->>'judge_reason' END AS judge_reason,
              -- Read through the row so this works before the import adds the column.
              COALESCE((to_jsonb(t)->>'attempt')::int, 1) AS attempt
         FROM bench_task_results t ORDER BY run_key, position`,
    );
    return { tasks: rows };
  });
}
