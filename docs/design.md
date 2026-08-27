# Design

## Contents

- [Principles](#principles)
- [Information architecture](#information-architecture)
- [Key interaction states](#key-interaction-states)
- [Responsive behaviour](#responsive-behaviour)
- [Accessibility](#accessibility)
- [Design decisions and why](#design-decisions-and-why)

---

## Principles

**1. The answer is the interface.** This is not a dashboard or a tool with
panels to learn — it is a conversation with a strong opinion about not lying to
you. Every other surface (sessions, artifacts, the provider badge) is
secondary and stays out of the way until it's needed.

**2. Provenance is not an afterthought.** A grounded assistant that hides its
sources is not meaningfully different from an ungrounded one. Citations are
always present on a grounded answer, always inspectable, never load-bearing
decoration.

**3. Untrusted content looks and behaves like untrusted content.** Generated
HTML renders in a visibly distinct panel with its own chrome, not inline in the
trusted chat surface. The user should never have to wonder whether something
on the page can act on their behalf.

**4. Local-model latency is designed for, not apologized for.** A CPU-only demo
answering in 10–30 seconds is not a bug to hide behind a generic spinner. The
UI treats "the assistant is thinking" as a real state with its own affordance.

---

## Information architecture

```
┌─ Sidebar (persistent, collapsible on mobile) ──────────────┐
│  Provider badge                                             │
│  + New chat                                                 │
│  Session list (title, message count)                        │
│  KB status footer (episodes/chunks indexed)                  │
└───────────────────────────────────────────────────────────┘

┌─ Main (chat) ───────────────┐  ┌─ Artifact panel (on demand) ─┐
│  KB-building banner          │  │  Title · kind badge           │
│  Error banner                │  │  Preview / Source tabs        │
│  Message stream               │  │  Sanitizer disclosure         │
│    - user bubbles             │  │  Rendered content              │
│    - assistant bubbles        │  │    (sandboxed iframe for HTML, │
│      + citations (collapsed)  │  │     react-markdown for MD)     │
│      + artifact chip          │  └────────────────────────────────┘
│  Thinking indicator           │
│  Composer + suggestions       │
└───────────────────────────────┘
```

Three panes, never more. The artifact panel only exists when there is an
artifact to show — it does not reserve empty space, because most turns in a
Q&A-heavy session never produce one.

---

## Key interaction states

| State | Treatment | Why |
|---|---|---|
| **Boot, no sessions yet** | Auto-creates one so the user lands in a composer, not an empty list | A blank "0 conversations" screen is a dead end, not a starting point |
| **Knowledge base still seeding** | Amber banner, persists across reloads via polling, disappears the instant `ready` flips | Given multi-hour ingestion is real on CPU-only hardware, the app must be honest that answers are provisional rather than silently letting the user think retrieval is complete |
| **Assistant thinking** | Three-dot pulse in a bubble shaped like the answer that's coming, `aria-live="polite"` | A local model can take 10–30s; a bare spinner reads as broken past ~5s, a bubble reads as "in progress" |
| **Grounded answer** | Citations collapsed under a `N sources` disclosure, one click to expand | Showing 6 source cards under every answer buries the answer; hiding them entirely defeats the point of grounding |
| **Ungrounded / refused** | Plain sentence stating the corpus doesn't cover it, no citations rendered (there are none) | A refusal that *looks* uncertain (hedging, apologizing at length) is worse than one that's simply direct |
| **Artifact created** | A chip appears under the message; clicking opens the panel; the panel auto-opens on creation | The model is instructed not to repeat artifact content in chat — the chip is the only way back to it if the user closes the panel |
| **Error** (any typed API error) | Red banner with the human message, the actionable hint, and the error code + request id | The hint is what makes it possible to *do something* — "start Ollama" beats "something went wrong" |
| **Empty session** | Centered prompt with three suggested questions | Cold-start problem: a blank composer with no examples under-communicates what this even does |

---

## Responsive behaviour

- **≥ 1024px (desktop):** sidebar and artifact panel both persistent alongside
  chat, three-column layout.
- **< 1024px (tablet/mobile):** sidebar becomes an overlay drawer (hamburger
  trigger, scrim, closes on selection). The artifact panel becomes a full-screen
  overlay rather than a third column — on a narrow screen, a side-by-side
  chat+artifact split leaves neither one usable.
- Composer textarea grows with content up to a capped height so a long question
  doesn't take over the viewport on a small screen.

---

## Accessibility

- **Keyboard.** A skip link jumps straight to the composer. Tab order follows
  visual order through sidebar → chat → artifact panel. Enter sends, Shift+Enter
  inserts a newline — the standard chat convention, not a novel one to learn.
- **Focus visibility.** A visible focus ring (`:focus-visible`) on every
  interactive element; never suppressed.
- **Live regions.** The thinking indicator and error banner are
  `aria-live="polite"` / `role="alert"` so a screen reader user learns the
  assistant is working or that something failed without hunting for it.
- **Semantic structure.** The artifact viewer's preview/source toggle uses
  `role="tablist"`/`aria-selected`; expandable citations use `aria-expanded`;
  icon-only buttons (close, delete, sidebar toggle) all carry `aria-label`.
- **Motion.** The thinking-dot animation respects `prefers-reduced-motion`,
  falling back to a static state rather than a forced pulse.
- **Color is never the only signal.** The provider badge pairs a dot with text
  ("Local · llama3.2:3b"), not color alone; the KB-building and error banners
  carry a text label, not just a background tint.

---

## Design decisions and why

**Citations collapsed by default, not hidden or always-expanded.** Tested both
extremes in reasoning about this: always-expanded turns every answer into a
wall of source cards and the actual answer stops being the visual center of the
message; hidden-unless-asked defeats grounding as a *visible* property of the
product, which is the entire point of building this on a transcript corpus
rather than a generic model. Collapsed-with-a-count is the middle: the user
always sees "6 sources" without reading them, and one click away when they want
to check.

**The artifact viewer has a Source tab, not just Preview.** For HTML artifacts
specifically, this doubles as the security disclosure surface — a user who
distrusts what rendered can read exactly what was sanitized and kept, in plain
text, with no risk of it executing.

**No streaming tokens in v1.** A genuinely useful streaming experience needs
work across the agent loop, the API contract, and the UI, and with a
forced-retrieval step in the loop, the first token is already late regardless
of whether the rest streams. The three-dot "thinking" state was chosen
deliberately as the higher-leverage fix for the same underlying problem
(local-model latency feels slow) at a fraction of the implementation cost —
documented as an explicit scope cut in [docs/PRD.md](PRD.md#scope), not an
oversight.

**Suggested prompts on an empty composer, not a static onboarding screen.**
Clicking a suggestion fills the composer rather than sending immediately, so
the user can edit before committing — useful given a local-model turn is not
free to retry casually.
