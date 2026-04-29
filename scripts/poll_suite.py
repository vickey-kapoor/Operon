"""Poll a benchmark suite until completion and print a live status table."""
import sys
import time
import urllib.request
import json

BASE = "http://127.0.0.1:8080"

def fetch(suite_id: str) -> dict:
    url = f"{BASE}/benchmark/suite/{suite_id}"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

def render(d: dict) -> None:
    tasks = d.get("tasks", [])
    done = d["completed"]
    total = d["total"]
    passed = d["passed"]
    print(f"\n[{time.strftime('%H:%M:%S')}] Suite {d['suite_id']} | status={d['status']} | {done}/{total} done | {passed} passed | rate={d['pass_rate']:.0%}")
    print(f"{'ID':<12} {'status':<10} {'stop_reason':<30} {'steps':>5} {'eff':>5} {'dur':>6}")
    print("-" * 75)
    for t in tasks:
        eff = f"{t['step_efficiency']:.2f}" if t.get("step_efficiency") is not None else "-"
        dur = f"{t['duration_seconds']:.0f}s" if t.get("duration_seconds") is not None else "-"
        sr = (t.get("stop_reason") or t.get("error") or "-")[:28]
        print(f"{t['task_id']:<12} {t['status']:<10} {sr:<30} {t['step_count']:>5} {eff:>5} {dur:>6}")

def main():
    suite_id = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    while True:
        try:
            d = fetch(suite_id)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] poll error: {e}")
            time.sleep(interval)
            continue
        render(d)
        if d["status"] in ("completed", "expired"):
            print("\nDone.")
            break
        time.sleep(interval)

if __name__ == "__main__":
    main()
