# Secrets the model can type without seeing them (KAN-103)

Experimental, off by default: `OPERON_SECRETS=on`.

You store a password once. The model writes `{{secret:NAME}}` as the text of a type action, and
the executor swaps in the real value at the last moment, inside its type call. The model, the
run files, the logs and the live view only ever hold the placeholder.

```
operon secrets set bank_pw     # asks for the value (hidden, twice); never a command-line argument
operon secrets list            # names only
operon secrets rm bank_pw
```

Names are 1 to 64 letters, digits, `_` or `-`.

## Where values are stored

One file per secret under `~/.operon/secrets/` (the `OPERON_CONFIG_DIR` folder).

- **Windows:** each value is encrypted with DPAPI (`CryptProtectData`, current-user scope, no
  prompt). Another Windows user, or a copy of the file on another machine, cannot read it. Any
  program running as you can.
- **Linux and macOS:** the file has mode `0600` in a `0700` folder. **It is not encrypted at
  rest.** Anyone who can read your files (root, a backup, a stolen disk) can read it. Use
  full-disk encryption, or do not store secrets there.

## What the model sees

With the flag on and at least one secret stored, every step carries an advisory hint (source
`secrets`) naming the available secrets and the `{{secret:NAME}}` syntax, and telling the model
never to ask the user to reveal a secret. No value is ever in a prompt.

## Where the value is filled in

Every executor that types does the swap in a local variable and leaves the `AgentAction` as it
was, so nothing that records the action can hold the value:

| Executor | Behaviour |
|---|---|
| Browser (`NativeBrowserExecutor`, batch children included) | swaps before the focus click; types the value |
| Windows desktop (`DesktopExecutor`) | same, typed with `SendInput` (`KEYEVENTF_UNICODE`, surrogate pairs handled): **never the clipboard** |
| Linux desktop (`LinuxDesktopExecutor`) | typed key by key with `pyautogui.write`, any length, never the clipboard. A secret with a character that has no key (an accent, a newline, an emoji) **fails closed** before any click |
| Remote guest (Windows Sandbox, Linux container) | the **host** swaps, because the guest has no vault; the value crosses to the guest over the token-protected HTTP connection. That connection is plain HTTP, so it is sent only to loopback (the container) or to the exact guest address the provider reported (the Sandbox's host-only IP); any other address, private ranges included, fails the action. The Sandbox provider accepts the address its guest wrote to `ready.json` only if it is a loopback, private or link-local IP literal; otherwise secrets fail closed. The request carries a `secret_filled` marker so the guest types the text as it is (no second placeholder lookup, no clipboard) and masks it in its reply. The host also masks the reply and puts the original action back before the loop sees it |
| OSWorld VM | the value goes only into the command sent to the VM (`pyautogui.write`); non-ASCII secrets fail closed instead of being pasted |

A secret never goes through the clipboard (no Win+V history, no cloud clipboard). As a second guard,
`read_clipboard` output is masked against every secret typed in the run.

An unknown name fails the action with `unknown secret NAME` and types nothing (no focus click
either). A bad name fails the same way.

**With the flag off, a placeholder fails the action ("secrets are off") instead of being typed.**
Typing the literal text `{{secret:pw}}` into a password field would silently submit a wrong
password, and could lock an account.

The human takeover typing during a takeover is not the model's action and is not substituted.

## Approval

Typing a secret always needs a person. The trust gate answers CONFIRM (category `account`, reason
"typing a secret needs approval") for every type action, batch children included, that uses a
placeholder, in **every** permission mode: `ask`, `risky` and `auto` all pause. `auto` mode cannot
skip it. The one exception: if `OPERON_TRUST_ALLOW_DOMAINS` is set and the current page's host is
on it, the secret is typed without a pause (an operator who allowlisted a login site has already
said where passwords may go). A desktop app has no page, so it always pauses. The approval
request shows the placeholder and, in a browser, the host the page is on (read even with no allowlist; "on an unknown site" when the URL cannot be read), never the value. A desktop app shows no site. In a batch, the leaf that types the secret is always the one the approval describes. When earlier steps of the batch run first (a navigation, a click) the shown host is only the page at approval time, so the text says it is re-checked before typing: right before the secret is typed the page is read again, and if its host is not the one that was shown (or none was known) the batch stops before that step, nothing is typed, and the model's next turn gets the secret type gated again with the right site. This holds with or without an allowlist, in the thin and full harness. The same re-check covers a single secret type: if the page moved to another host while the run waited for approval, the run pauses again naming the new host instead of typing. A policy-blocked (DENY) action that also types a secret names the secret and site in its text too. A desktop app has no page, so none of these host checks apply there.

With the trust gate off (`OPERON_TRUST_GATE=off`, as in the benchmarks) a secret type **fails** with
"secrets need the trust gate on", so a secret is never typed with no approval step at all.

## Limits: read these

- **Screenshots show whatever the app renders.** Password fields are masked by the application,
  so the screenshot shows dots. A **plain text field shows the value**, and that screenshot goes
  to the model, is saved in the run folder, and appears in the live view. Only type secrets into
  fields the app masks. Operon cannot tell a masked field from a plain one.
- An app that echoes a password somewhere else (a "show password" toggle, an error message, a URL)
  puts it on screen the same way.
- The model can press a "show password" control itself. Keep `ask` mode for sensitive sites.
- Values live in memory while a type runs, and the executor keeps the values typed in a run for its
  clipboard mask until that run is cleaned up (or the executor closes). They never go on the clipboard.
- On an OSWorld VM the value is part of the `python3 -c` command sent to the VM, so it appears in
  that process's argv (visible to other processes on the VM while it runs). The VM URL
  (`OPERON_OSWORLD_URL`) must be loopback, or a secret type fails.
- The guest process (sandbox or container) holds the value while it types.
- The vault is only as safe as your user account. It is not a replacement for a password manager.
