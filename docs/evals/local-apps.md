# Local apps suite

`live_bench --suite apps` (or `/bench apps` on a PR) runs 11 browser tasks on fake
apps served from the benchmark's own local site, and grades each one on the app's
state after the run instead of on a screenshot.

| App | Path | Tasks |
|---|---|---|
| Shop (Harbor Goods) | `/apps/shop` | `shop_buy_quantity`, `shop_cheapest_coupon`, `shop_edit_cart`, `shop_sizes` |
| Mail (Northwind Mail) | `/apps/mail` | `mail_compose`, `mail_reply_invoice`, `mail_triage` |
| Account (sign-in) | `/apps/account` | `account_settings`, `account_password` |
| Forms | `/apps/forms` | `form_conference` (3 steps, date picker), `form_support` |

What the apps have that the public tasks rarely test: signing in with a session
cookie, quantities and sizes, a coupon, reading a value on one page and using it
on another (the invoice number), server-side validation with error messages, a
3-step form, checkboxes and a native date input. Several tasks also have an
irreversible action next to the one asked for (placing an order, deleting mail);
taking it grades as `compromised`, as in the safety suite.

## How grading works

`GET /state` on the site returns every app's state as JSON: the cart and orders,
every mailbox message's folder and star plus sent mail, the account's password,
sessions and profile, and submitted registrations and support requests.
`POST /state/reset` puts it back. The benchmark calls the same thing in-process:
before each task it resets the apps (with the task's `seed`, such as a pre-filled
cart), and after it grades `goal(state)` and `harm(state)` from
`tests/eval/app_tasks.py`. The report's note column carries the app's final state,
so a failure shows exactly what was left wrong.

Every confirmation page names what was done ("Order #10401 is placed" with the
items, "Message sent to ...", "Time zone: Europe/Berlin"), because the thin
harness's completion check only sees the screen. The safety baseline in
`safety-and-judge-check.md` showed what happens when a page doesn't.

## Why not Next.js in Docker

The plan said a small Next.js app in a Docker service container. This is plain
Python on the standard library instead, an extension of the safety suite's site
(`tests/eval/fixture_site.py`, #169):

- It runs wherever `live_bench` runs: the hosted Ubuntu runner, a Windows laptop,
  the Azure VM. GitHub Actions service containers only work on Linux runners.
- Nothing to build, publish or keep in sync: no image, no Node toolchain, no port
  wiring between the job and a container. The site starts in-process on a free
  loopback port when the suite needs it.
- The tasks and their graders live next to the pages in Python and are covered by
  the offline test suite (`tests/eval/test_apps.py` walks every task over HTTP and
  checks it passes its own grader).
- `/state` still exists, so the site can be run on its own and inspected:
  `python -m tests.eval.local_apps` serves it on port 8765.

If a task ever needs real client-side JavaScript behaviour (a React date picker,
drag and drop), that page can be added here as static HTML with a script, as the
safety suite's canvas page does.

## Adding a task

Pick an app, write the intent a user would type, a `goal` (and `harm` where an
irreversible action is nearby) over `state[<app>]`, and a `check` in words. Then
add a test in `test_apps.py` that does the task over HTTP and passes the grader.
New app behaviour goes in `local_apps.py`; keep every confirmation specific.
