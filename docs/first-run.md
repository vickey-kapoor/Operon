# First run, step by step

This page walks a first-time user from nothing to a finished Operon task with
`operon-agent` 0.1.0 from PyPI. It's the script for the stranger test (Jira KAN-142).
It should take about 5 minutes.

Start a timer at step 1. Run each command on its own and wait for it to finish
before starting the next one. Write down anything that confuses you.

## Before you start

- **Python 3.14 or newer.** Check it with `python --version`.
- **A free Gemini API key** from <https://aistudio.google.com/apikey>. Sign in with a
  Google account, click "Create API key" and keep the key ready for step 5.

## Step 1: make an empty folder and open a terminal in it

Any folder works, for example `operon-test` on your Desktop. Use PowerShell on
Windows, or Terminal on Mac and Linux.

## Step 2: create a private Python environment

```bash
python -m venv .venv
```

- `python -m venv` is Python's built-in tool for making a private environment.
- `.venv` is the name of the folder it creates in your current folder.
- **Changes:** adds a `.venv` folder and nothing else.
- **Cost:** free.
- **Expected output:** none. The command returns after a few seconds.

## Step 3: switch to that environment

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Mac or Linux:

```bash
source .venv/bin/activate
```

- This makes `python` and `pip` use the private environment, in this terminal only.
- **Changes:** nothing on disk.
- **Cost:** free.
- **Expected output:** your prompt now starts with `(.venv)`.
- If Windows says running scripts is disabled, run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (this window only),
  then run the activate command again.

## Step 4: install Operon from PyPI

```bash
pip install operon-agent
```

- `pip install` downloads a package from PyPI, the public Python package index.
- `operon-agent` is Operon's package name there.
- **Changes:** installs Operon and its libraries into `.venv` only.
- **Cost:** free. It takes about a minute.
- **Expected output:** the last line starts with `Successfully installed` and
  includes `operon-agent-0.1.0`.

## Step 5: set up Operon

```bash
operon setup
```

- `operon` is the command that step 4 installed, and `setup` is its one-time setup.
- When asked, paste your Gemini key and press Enter. The key stays hidden while you
  paste it.
- It then:
  - checks the key with Gemini
  - saves it in a private settings folder, `~/.operon`, that only your user can read
  - downloads the Chromium browser that Operon drives
  - runs a self-check
- **Changes:** creates `~/.operon` and puts a copy of Chromium in your user folder.
- **Cost:** free, apart from one tiny key check. It takes about two minutes.
- **Expected output:** a list of checks ending with `Ready.` If a line says `FAIL`,
  it says what to fix. Fix it and run `operon setup` again.

## Step 6: run a real task

```bash
operon run "Find the population of Paris on Wikipedia" --url https://en.wikipedia.org
```

- `operon run` starts one task.
- The text in quotes is the task, in plain English.
- `--url` is the page the browser opens first.
- A browser window opens and Operon works through the task from screenshots alone,
  printing each step. Don't touch the browser while it runs.
- If it stops to ask about a risky step, type `y` to allow it, `n` to refuse it, or
  `t` to take over. This task shouldn't need any of them. Ctrl+C stops it at any time.
- **Changes:** saves the run's screenshots and log under `~/.operon`.
- **Cost:** a few cents of Gemini calls on your own key (about $0.04 in our tests).
- **Expected output:** after about 5 steps and 30 to 90 seconds, a short report with
  the answer, which is about 2.1 million.

Stop the timer.

## If Windows blocks `operon.exe` or `pip.exe`

Smart App Control can block these programs. Run the same commands through Python
instead:

| Step | Command |
| --- | --- |
| 4 | `python -m pip install operon-agent` |
| 5 | `python -m operon.cli setup` |
| 6 | `python -m operon.cli run "Find the population of Paris on Wikipedia" --url https://en.wikipedia.org` |

## Tell us how it went

Report back as a comment on KAN-142, or in a GitHub issue:

- your total time, from step 1 to the answer
- your OS and Python version
- whether it finished with the right answer
- every place you got stuck or confused, with any error text

## Clean up (optional)

Delete the test folder and `~/.operon`.
