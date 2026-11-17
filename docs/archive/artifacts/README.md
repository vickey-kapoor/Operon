# Archived claude.ai artifacts

Exports of the Operon artifacts that were published on claude.ai, saved here on 2026-11-17 so the
artifacts themselves can be deleted. Content is kept as exported; nothing was edited.

| Folder / file | Was | Type | Last updated |
| --- | --- | --- | --- |
| `windows-mockups/` | Operon for Windows mockups (v2, "Your team", 12 screens) | Design canvas | 2026-11-11 |
| `desktop-key-screens/` | Operon Desktop: key screens (desktop launcher + local web version, 7 screens) | Design canvas | 2026-11-11 |
| `grok-bot-ideas.md` | Operon Desktop: ideas inspired by Grok Bot | Doc | 2026-11-11 |
| `scaling-roadmap-slides/` | Operon scaling roadmap (10 slides) | Slide deck | 2026-11-14 |
| `scaling-roadmap-whiteboard/board.json` | Operon scaling roadmap | Whiteboard | 2026-11-14 |

Notes on the formats:

- The design canvases are one `*.dc.html` file per screen plus `canvas.json` (layout and titles).
  The screens are plain HTML with inline styles; they reference a `support.js` runtime that only
  exists inside the claude.ai editor, so open them in a browser for a rough look, not a pixel match.
- The slide deck is `deck.json` (order and sections), `deck.css` (shared styles) and one
  `slides/<id>.html` per slide, with speaker notes in each slide's `<aside>`.
- The whiteboard is its element list (shapes, text, arrows with x/y positions) in drawing order.

Older plan artifacts in the project list (Q4 2026 plan, next development plan, evals and expansion,
expansion plan, eval expansion plan, cross-model benchmark design, architecture review, security
audit, and an earlier copy of the Windows mockups) were already gone when this export ran, so there
was nothing left to save from them. Several of those plans already live in `docs/`
(`expansion-plan.md`, `eval-expansion-plan.md`, `cross-model-benchmark-proposal.md`, `roadmap.md`).
