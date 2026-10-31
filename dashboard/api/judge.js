// GET /api/judge: the judge's agreement with hand labels, one row per check, oldest first.
// evals/judge_check.py writes the judge_checks table; before its first save there is none.
import { db, handle } from "../server/db.js";

export default function handler(req, res) {
  return handle(res, async () => {
    const exists = await db().query("SELECT to_regclass('judge_checks') IS NOT NULL AS ok");
    if (!exists.rows[0].ok) return { checks: [] };
    const { rows } = await db().query(
      `SELECT checked_at, judge_model, labels, repeats, accuracy, precision, recall, agreement,
              false_passes, false_fails, disagreements
         FROM judge_checks ORDER BY checked_at`,
    );
    return { checks: rows };
  });
}
