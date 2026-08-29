"""One-off script that renders docs/architecture_v6.png.

Structured after the two reference diagrams: a "Harness" (the live agent loop
plus the memory that feeds it) beside an "LLM Ops" governance loop
(trace -> eval -> observe -> diagnose -> gate -> release), over a shared data
layer.

Mapped onto what this project actually has, not the reference's exact boxes:
  - No semantic memory (durable per-user facts) and no consolidation/summariser
    step exist here -- called out in the diagram rather than invented.
  - Episodic memory is full, unsummarised history in Postgres.
  - A Knowledge Base box is added, because retrieval here is over a fixed
    303-episode transcript corpus reached only through `search_transcripts`.
  - "LLM Ops" is the two eval harnesses + the trace store + agent-transcripts/.

Every string drawn inside a box is width-checked against that box at render
time (see `fit`); the script prints violations instead of silently clipping,
because the previous revision shipped text that overflowed its containers.

Run with: python render_architecture_v6.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1940, 1110

BG = (250, 247, 237)
INK = (35, 35, 40)
INK_SOFT = (90, 90, 98)
INK_MUTED = (140, 140, 146)

RED = (176, 42, 34)
RED_SOFT = (253, 240, 237)
BLUE = (36, 66, 145)
BLUE_SOFT = (235, 240, 252)
AMBER = (168, 100, 12)
AMBER_SOFT = (255, 246, 227)
GREEN = (35, 110, 60)
GREEN_SOFT = (232, 247, 235)
PINK = (176, 50, 110)
PINK_SOFT = (252, 235, 244)
TEAL = (16, 110, 110)
TEAL_SOFT = (228, 246, 246)
GREY_BORDER = (150, 150, 145)
WHITE = (255, 255, 255)

FONT_DIR = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + name, size)


F_TITLE = font("comicbd.ttf", 34)
F_SUBTITLE = font("comic.ttf", 16)
F_HDR_TAG = font("consolab.ttf", 13)
F_META = font("consola.ttf", 12)
F_SECTION = font("comicbd.ttf", 18)
F_SECTION_SUB = font("consola.ttf", 12)
F_BOX_TITLE = font("comicbd.ttf", 14)
F_BOX_SUB = font("consola.ttf", 11)
F_CAPTION = font("consola.ttf", 12)
F_LABEL = font("consola.ttf", 11)

_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
VIOLATIONS: list[str] = []


def measure(text: str, fnt) -> float:
    return _probe.textlength(text, font=fnt)


def fit(text: str, fnt, max_w: float, where: str) -> None:
    """Record (don't silently clip) any string wider than its container."""
    w = measure(text, fnt)
    if w > max_w:
        VIOLATIONS.append(f"{where}: {w:.0f}px > {max_w:.0f}px :: {text[:70]}")


def rrect(d, box_, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box_, radius=r, fill=fill, outline=outline, width=width)


def dashed_rect(d, box_, color, width=2, dash_len=6, gap_len=4):
    x0, y0, x1, y1 = box_
    for p1, p2 in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                   ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        _dashed_line(d, p1, p2, color, width, dash_len, gap_len)


def _dashed_line(d, p1, p2, color, width, dash_len, gap_len):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos, draw_seg = 0.0, True
    while pos < length:
        seg = dash_len if draw_seg else gap_len
        end = min(pos + seg, length)
        if draw_seg:
            d.line([(x1 + dx * pos, y1 + dy * pos), (x1 + dx * end, y1 + dy * end)],
                   fill=color, width=width)
        pos = end
        draw_seg = not draw_seg


def ctext(d, xy, text, fnt, fill, anchor="mm"):
    d.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def ltext(d, xy, text, fnt, fill):
    d.text(xy, text, font=fnt, fill=fill, anchor="lm")


def arrowhead(d, tip, angle, color, size=7):
    left = (tip[0] - size * math.cos(angle - 0.45), tip[1] - size * math.sin(angle - 0.45))
    right = (tip[0] - size * math.cos(angle + 0.45), tip[1] - size * math.sin(angle + 0.45))
    d.polygon([tip, left, right], fill=color)


def arrow(d, p1, p2, color=INK_MUTED, width=2):
    d.line([p1, p2], fill=color, width=width)
    arrowhead(d, p2, math.atan2(p2[1] - p1[1], p2[0] - p1[0]), color)


def polyline(d, pts, color=INK_MUTED, width=2):
    """Orthogonal multi-segment connector with a head on the final segment."""
    d.line(pts, fill=color, width=width)
    a, b = pts[-2], pts[-1]
    arrowhead(d, b, math.atan2(b[1] - a[1], b[0] - a[0]), color)


def box(d, rect, title, sub=None, fill=WHITE, border=GREY_BORDER, title_color=INK, tag=""):
    rrect(d, rect, 8, fill=fill, outline=border, width=2)
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    avail = (x1 - x0) - 18
    fit(title, F_BOX_TITLE, avail, tag or title)
    if isinstance(sub, tuple):
        ctext(d, (cx, cy - 14), title, F_BOX_TITLE, title_color)
        for i, line in enumerate(sub):
            fit(line, F_BOX_SUB, avail, tag or title)
            ctext(d, (cx, cy + 2 + i * 14), line, F_BOX_SUB, INK_MUTED)
    elif sub:
        fit(sub, F_BOX_SUB, avail, tag or title)
        ctext(d, (cx, cy - 8), title, F_BOX_TITLE, title_color)
        ctext(d, (cx, cy + 9), sub, F_BOX_SUB, INK_MUTED)
    else:
        ctext(d, (cx, cy), title, F_BOX_TITLE, title_color)


def diamond(d, cx, cy, w, h, title, sub, color, fill):
    d.polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)],
              fill=fill, outline=color, width=2)
    fit(sub, F_BOX_SUB, w * 0.62, "gate")
    ctext(d, (cx, cy - 7), title, F_BOX_TITLE, color)
    ctext(d, (cx, cy + 8), sub, F_BOX_SUB, INK_MUTED)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # ---------------- Header ----------------
    ctext(d, (36, 38), "The Lenny Growth Assistant", F_TITLE, INK, anchor="lm")
    ctext(d, (36, 70),
          "The shape of any agent system: a loop with memory, watched by a separate ops loop.",
          F_SUBTITLE, INK_SOFT, anchor="lm")
    ctext(d, (W - 36, 26), "LENNY GROWTH ASSISTANT \u2014 V4 ARCHITECTURE", F_HDR_TAG, RED, anchor="rm")
    ctext(d, (W - 36, 44), "ADR-0004 \u00b7 status: shipped \u00b7 2026-08-28", F_META, INK_MUTED, anchor="rm")

    PANEL_TOP, PANEL_BOT = 120, 945

    # ================= LEFT: HARNESS =================
    rrect(d, (40, PANEL_TOP, 1150, PANEL_BOT), 16, fill=RED_SOFT, outline=RED, width=3)
    ltext(d, (62, 144), "HARNESS \u2014 Agent Loop", F_SECTION, RED)
    ltext(d, (62, 165), "run_agent()  \u00b7  everything above the memory row is one chat turn",
          F_SECTION_SUB, INK_MUTED)

    # --- inputs ---
    box(d, (62, 190, 300, 235), "User Prompt", fill=GREEN_SOFT, border=GREEN)
    box(d, (62, 245, 300, 290), "Current chat history")
    box(d, (62, 300, 300, 345), "System prompt")

    wm = (340, 230, 560, 310)
    box(d, wm, "Working Memory", ("app/memory/reducers.py", "builds this turn's message list"),
        fill=BLUE_SOFT, border=BLUE)
    arrow(d, (302, 212), (336, 255))
    arrow(d, (302, 267), (336, 268))
    arrow(d, (302, 322), (336, 285))

    # --- the model ---
    ecx, ecy, ehw, ehh = 700, 270, 90, 65
    d.ellipse([ecx - ehw, ecy - ehh, ecx + ehw, ecy + ehh], outline=PINK, width=3, fill=PINK_SOFT)
    for dy, txt, f, col in ((-22, "LLM Q&A Agent", F_BOX_TITLE, PINK),
                            (-4, "Ollama (local) or", F_BOX_SUB, INK_SOFT),
                            (14, "OpenAI-compat (cloud)", F_BOX_SUB, INK_SOFT)):
        fit(txt, f, 2 * ehw * math.sqrt(max(1 - (dy / ehh) ** 2, 0.01)) - 12, "llm ellipse")
        ctext(d, (ecx, ecy + dy), txt, f, col)
    arrow(d, (562, 270), (606, 270))

    # --- loop ---
    rrect(d, (850, 180, 1140, 340), 12, fill=AMBER_SOFT, outline=AMBER, width=3)
    ltext(d, (868, 202), "Loop", F_SECTION, AMBER)
    box(d, (866, 228, 1124, 322), "Tool registry",
        ("search \u00b7 ship30 essay \u00b7 artifact", "only 3 tools, on purpose"),
        fill=WHITE, border=AMBER)
    arrow(d, (793, 250), (846, 250), color=AMBER)
    ctext(d, (820, 238), "tools", F_LABEL, AMBER)
    arrow(d, (846, 292), (793, 292), color=AMBER)
    ctext(d, (820, 280), "reply", F_LABEL, AMBER)

    # --- guardrails ---
    guard = (340, 380, 900, 480)
    dashed_rect(d, guard, RED, 2, 6, 4)
    gl = ["End-Loop Guardrails \u2014 6 deterministic rules",
          "force search \u00b7 block ungrounded artifact \u00b7 block redundant artifact",
          "force artifact when asked \u00b7 no tools on small talk",
          "no fabricated citations \u2014 each found live, see agent-transcripts/"]
    for i, line in enumerate(gl):
        fit(line, F_BOX_SUB, 544, "guardrails")
        ctext(d, (620, 400 + i * 16), line, F_BOX_SUB, RED)
    arrow(d, (700, 336), (700, 378))

    box(d, (480, 515, 760, 575), "Reply",
        ("+ citations \u00b7 + artifact", "grounded in transcripts, not guessed"),
        fill=GREEN_SOFT, border=GREEN)
    arrow(d, (620, 482), (620, 513))

    # --- memory row ---
    mem_y, mem_h = 608, 122
    mw = (1140 - 62 - 48) / 3
    proc = (62, mem_y, 62 + mw, mem_y + mem_h)
    epis = (62 + mw + 24, mem_y, 62 + 2 * mw + 24, mem_y + mem_h)
    kb = (62 + 2 * mw + 48, mem_y, 1140, mem_y + mem_h)
    box(d, proc, "Procedural Memory",
        ("memory/procedural.py", "one principle, injected into the system prompt"),
        border=TEAL)
    box(d, epis, "Episodic Memory",
        ("Postgres: sessions + messages, in full", "not summarised \u2014 nothing is distilled"),
        border=TEAL)
    box(d, kb, "Knowledge Base (RAG)",
        ("Postgres + pgvector, 303 episodes", "reached only via search_transcripts"),
        border=TEAL)

    # Procedural + episodic join a bus and rise through the clear left corridor.
    # A straight diagonal from either box to Working Memory would cut through
    # the guardrail box and the Reply box, so the path is routed, not direct.
    d.line([(601, mem_y - 2), (601, 588), (233, 588)], fill=TEAL, width=2)
    polyline(d, [(233, mem_y - 2), (233, 358), (420, 358), (420, 312)], color=TEAL)
    ctext(d, (150, 450), "memory feeds", F_LABEL, TEAL)
    ctext(d, (150, 464), "working memory", F_LABEL, TEAL)

    arrow(d, (968, mem_y - 2), (1000, 342), color=TEAL)
    ctext(d, (1012, 470), "hybrid search:", F_LABEL, TEAL, anchor="lm")
    ctext(d, (1012, 484), "top-k + FTS", F_LABEL, TEAL, anchor="lm")

    note = (62, 748, 1140, 806)
    dashed_rect(d, note, TEAL, 2, 6, 4)
    for i, line in enumerate([
        "No semantic memory (durable per-user facts) and no consolidation/summariser step exist here \u2014",
        "left out rather than faked. Episodic memory stays exactly as recorded."]):
        fit(line, F_BOX_SUB, 1062, "memory note")
        ctext(d, (601, 768 + i * 16), line, F_BOX_SUB, TEAL)

    box(d, (62, 826, 560, 898), "Execution Trace",
        ("memory/trace.py \u2014 separate SQLite db", "one row per LLM/tool call, timing + tokens"))
    cap = "\u2191 written during the Loop above \u2014 a one-way record, not something the model reads back"
    fit(cap, F_CAPTION, 1080, "trace caption")
    ltext(d, (62, 918), cap, F_CAPTION, INK_MUTED)

    # ================= RIGHT: LLM OPS =================
    rrect(d, (1200, PANEL_TOP, 1900, PANEL_BOT), 16, fill=BLUE_SOFT, outline=BLUE, width=3)
    ltext(d, (1224, 144), "LLM OPS \u2014 Trace, Eval, Diagnose, Gate, Release", F_SECTION, BLUE)
    ltext(d, (1224, 165), "app/evals/ + memory/trace.py + agent-transcripts/", F_SECTION_SUB, INK_MUTED)

    ox0, ox1 = 1270, 1875
    cx = (ox0 + ox1) / 2

    box(d, (ox0, 195, ox1, 255), "Trace",
        ("1 row per LLM / tool call", "from the Execution Trace store (left)"), border=BLUE)

    rrect(d, (ox0, 275, ox1, 375), 10, fill=WHITE, outline=BLUE, width=2)
    ctext(d, (cx, 293), "Eval \u2014 two harnesses", F_BOX_TITLE, BLUE)
    box(d, (1286, 310, 1568, 365), "LLM Harness", "24 golden Qs, retrieval only",
        fill=BLUE_SOFT, border=BLUE)
    box(d, (1578, 310, 1859, 365), "Agent Harness", "8 scenarios, real run_agent()",
        fill=BLUE_SOFT, border=BLUE)
    arrow(d, (cx, 256), (cx, 273), color=BLUE)

    box(d, (ox0, 395, ox1, 455), "Observe",
        ("structured logs (app.logging) + /health/deep", "was the run healthy, not just correct?"),
        border=BLUE)
    arrow(d, (cx, 376), (cx, 393), color=BLUE)

    box(d, (ox0, 475, ox1, 545), "Diagnose",
        ("agent-transcripts/ \u2014 13 entries", "root cause written down before any fix"),
        border=BLUE)
    arrow(d, (cx, 456), (cx, 473), color=BLUE)

    diamond(d, cx, 610, 220, 90, "Gate", "score \u2265 threshold?", BLUE, WHITE)
    arrow(d, (cx, 546), (cx, 563), color=BLUE)

    box(d, (ox0, 690, ox1, 770), "Release",
        ("commit + push, new agent-transcripts entry",
         "a prompt/config change ships with its own record"),
        fill=GREEN_SOFT, border=GREEN)
    arrow(d, (cx, 657), (cx, 688), color=GREEN)
    ctext(d, (cx + 34, 673), "pass", F_LABEL, GREEN)

    # fail path -- stays entirely inside the ops panel
    polyline(d, [(cx - 110, 610), (1235, 610), (1235, 320), (ox0 - 3, 320)], color=RED)
    lab = "fail \u2014 fix, re-run"
    fit(lab, F_LABEL, 170, "fail label")
    ctext(d, (1352, 596), lab, F_LABEL, RED)

    for i, line in enumerate([
        "found live via this exact loop: the 0.55 grounding floor,",
        "3 hallucinations, 2 small-talk routing bugs \u2014 agent-transcripts/10\u201313"]):
        fit(line, F_CAPTION, 605, "ops caption")
        ctext(d, (cx, 866 + i * 16), line, F_CAPTION, INK_SOFT)

    # --- trace feeds the ops loop (single clean border crossing) ---
    polyline(d, [(562, 862), (1180, 862), (1180, 225), (ox0 - 3, 225)], color=INK_MUTED)
    ltext(d, (700, 850), "every run's spans feed the ops loop", F_LABEL, INK_MUTED)

    # ---------------- Data layer ----------------
    ltext(d, (40, 968), "DATA LAYER  \u2014  one Postgres instance backs both the Knowledge Base and Episodic Memory above",
          F_SECTION_SUB, INK_MUTED)
    dw = (1860 - 64) / 3
    box(d, (40, 990, 40 + dw, 1070), "Postgres + pgvector",
        ("episodes \u00b7 chunks \u00b7 sessions \u00b7 messages \u00b7 artifacts",
         "one database, two logical uses shown above"))
    box(d, (72 + dw, 990, 72 + 2 * dw, 1070), "Ingestion",
        ("clone \u2192 chunk \u2192 embed \u2192 upsert, 303 episodes",
         "idempotent \u2014 safe to stop and resume anytime"))
    box(d, (104 + 2 * dw, 990, 1900, 1070), "Model config",
        ("LLM_PROVIDER=ollama | openai_compat",
         "one env var switches local vs. cloud model"))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "docs", "architecture_v6.png")
    img.save(out)
    print(f"saved {out} ({img.size[0]}x{img.size[1]})")
    if VIOLATIONS:
        print(f"\n{len(VIOLATIONS)} TEXT-FIT VIOLATION(S):")
        for v in VIOLATIONS:
            print("  " + v)
    else:
        print("text fit: OK (every string fits its container)")


if __name__ == "__main__":
    main()
