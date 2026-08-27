"""One-off script that renders docs/architecture_v2.png.

A hand-drawn-style "harness + bench" diagram, in the same visual language as
the reference ADR-style diagrams the user shared: marker-style titles,
monospace annotations, colored-border subsystem boxes, a two-column
harness/bench layout, a stack row, and a closing "what this fixes" narrative.

Not part of the running application -- run with:
    python render_architecture_v2.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1720, 1370

BG = (250, 247, 237)  # warm cream, matching the reference
INK = (35, 35, 40)
INK_SOFT = (80, 80, 88)
INK_MUTED = (130, 130, 138)
MONO = (90, 90, 98)

RED = (176, 42, 34)
RED_SOFT = (253, 240, 237)
BLUE = (36, 66, 145)
BLUE_SOFT = (235, 240, 252)
AMBER = (168, 100, 12)
AMBER_SOFT = (255, 246, 227)
GREEN = (35, 110, 60)
GREEN_SOFT = (232, 247, 235)
GREY_BORDER = (150, 150, 145)
WHITE = (255, 255, 255)

FONT_DIR = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + name, size)


F_TITLE = font("comicbd.ttf", 40)
F_SUBTITLE = font("comic.ttf", 18)
F_TAGLINE = font("consola.ttf", 15)
F_HDR_TAG = font("consolab.ttf", 13)
F_META = font("consola.ttf", 13)
F_SECTION = font("comicbd.ttf", 20)
F_SECTION_SUB = font("comic.ttf", 13)
F_BOX_TITLE = font("comicbd.ttf", 16)
F_BOX_MONO = font("consola.ttf", 12)
F_BOX_MONO_B = font("consolab.ttf", 12)
F_SMALL_MONO = font("consola.ttf", 11)
F_PILL = font("consolab.ttf", 13)
F_PILL_SUB = font("consola.ttf", 11)
F_BODY = font("comic.ttf", 15)


def rrect(d, box, r, fill=None, outline=None, width=2, dash=False):
    if dash and outline:
        x0, y0, x1, y1 = box
        # simple dashed rounded rect via dashed straight segments (corners left solid-ish)
        d.rounded_rectangle(box, radius=r, fill=fill, outline=None)
        _dashed_rect_outline(d, box, outline, width)
    else:
        d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def _dashed_rect_outline(d, box, color, width, dash_len=7, gap_len=5):
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for p1, p2 in edges:
        _dashed_line(d, p1, p2, color, width, dash_len, gap_len)


def _dashed_line(d, p1, p2, color, width, dash_len=7, gap_len=5):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    draw_seg = True
    while pos < length:
        seg = dash_len if draw_seg else gap_len
        end = min(pos + seg, length)
        if draw_seg:
            d.line([(x1 + dx * pos, y1 + dy * pos), (x1 + dx * end, y1 + dy * end)], fill=color, width=width)
        pos = end
        draw_seg = not draw_seg


def ctext(d, xy, text, fnt, fill, anchor="mm"):
    d.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def ltext(d, xy, text, fnt, fill):
    d.text(xy, text, font=fnt, fill=fill, anchor="lm")


def wrap_and_draw_left(d, xy, text, fnt, fill, max_width, line_h):
    x, y = xy
    words = text.split(" ")
    line = ""
    for w in words:
        trial = (line + " " + w).strip()
        if d.textlength(trial, font=fnt) > max_width and line:
            d.text((x, y), line, font=fnt, fill=fill, anchor="lm")
            y += line_h
            line = w
        else:
            line = trial
    if line:
        d.text((x, y), line, font=fnt, fill=fill, anchor="lm")
    return y + line_h


def arrowhead(d, tip, angle, color, size=8):
    left = (tip[0] - size * math.cos(angle - 0.45), tip[1] - size * math.sin(angle - 0.45))
    right = (tip[0] - size * math.cos(angle + 0.45), tip[1] - size * math.sin(angle + 0.45))
    d.polygon([tip, left, right], fill=color)


def arrow(d, p1, p2, color=INK_MUTED, width=2):
    d.line([p1, p2], fill=color, width=width)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    arrowhead(d, p2, angle, color)


def elbow(d, p1, p2, color=INK_MUTED, width=2, via="h"):
    mid = (p2[0], p1[1]) if via == "h" else (p1[0], p2[1])
    d.line([p1, mid], fill=color, width=width)
    d.line([mid, p2], fill=color, width=width)
    angle = math.atan2(p2[1] - mid[1], p2[0] - mid[0])
    arrowhead(d, p2, angle, color)


def note_box(d, box, title, lines, title_color, fill=WHITE, border=GREY_BORDER, mono=True):
    rrect(d, box, 8, fill=fill, outline=border, width=2)
    x0, y0, x1, y1 = box
    ctext(d, (x0 + 14, y0 + 20), title, F_BOX_TITLE, title_color, anchor="lm")
    y = y0 + 40
    fnt = F_BOX_MONO if mono else F_BODY
    for line in lines:
        d.text((x0 + 14, y), line, font=fnt, fill=INK_SOFT, anchor="lm")
        y += 18


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # ---- Header ----
    ctext(d, (40, 44), "The Lenny Growth Assistant", F_TITLE, INK, anchor="lm")
    ctext(
        d, (40, 82),
        "A grounded RAG assistant over Lenny's Podcast \u2014 gated by its own eval harness before it ships.",
        F_SUBTITLE, INK_SOFT, anchor="lm",
    )
    ctext(d, (W - 40, 30), "LENNY GROWTH ASSISTANT \u2014 V1 ARCHITECTURE", F_HDR_TAG, RED, anchor="rm")
    ctext(d, (W - 40, 50), "ADR-0001 \u00b7 status: shipped", F_META, INK_MUTED, anchor="rm")
    ctext(d, (W - 40, 68), "2026-08-27", F_META, INK_MUTED, anchor="rm")

    ctext(
        d, (W / 2, 112),
        "hybrid retrieval + deterministic guards + measured grounding \u2014 the loop closes here",
        F_TAGLINE, GREEN, anchor="mm",
    )

    # ---- Outer dashed container ----
    outer = (30, 132, W - 30, 900)
    _dashed_rect_outline(d, outer, GREY_BORDER, 2, dash_len=6, gap_len=5)

    # ================= LEFT: CHAT / AGENT LOOP (red) =================
    left = (55, 155, 850, 878)
    rrect(d, left, 14, fill=RED_SOFT, outline=RED, width=3)
    ctext(d, (75, 175), "CHAT \u2014 Agent Loop (per turn)", F_SECTION, RED, anchor="lm")
    ctext(d, (75, 198), "one run_agent() call per chat turn \u00b7 tool-calling loop, max 5 iterations", F_SECTION_SUB, INK_MUTED, anchor="lm")

    browser = (75, 225, 330, 285)
    note_box(d, browser, "Browser", ["chat + artifact viewer"], INK)

    nextjs = (360, 225, 610, 300)
    note_box(d, nextjs, "Next.js UI", ["sessions \u00b7 citations", "provider badge"], INK)
    arrow(d, (330, 255), (358, 255))

    bridge = (640, 225, 830, 300)
    note_box(d, bridge, "FastAPI /chat", ["persists user msg", "before agent runs"], RED)
    arrow(d, (610, 262), (638, 262))

    loop_circle_c = (500, 400)
    loop_r = 70
    d.ellipse(
        [loop_circle_c[0] - loop_r, loop_circle_c[1] - loop_r, loop_circle_c[0] + loop_r, loop_circle_c[1] + loop_r],
        outline=AMBER, width=3, fill=AMBER_SOFT,
    )
    ctext(d, (loop_circle_c[0], loop_circle_c[1] - 12), "Router / Agent", F_BOX_TITLE, AMBER, anchor="mm")
    ctext(d, (loop_circle_c[0], loop_circle_c[1] + 10), "3 deterministic", F_BOX_MONO, INK_SOFT, anchor="mm")
    ctext(d, (loop_circle_c[0], loop_circle_c[1] + 26), "guards", F_BOX_MONO, INK_SOFT, anchor="mm")
    elbow(d, (735, 300), (500, 330), via="v")

    tools = (610, 355, 830, 460)
    note_box(
        d, tools, "Tool registry",
        ["search_transcripts()", "write_ship30_essay()", "create_artifact()"],
        GREEN, fill=GREEN_SOFT, border=GREEN,
    )
    arrow(d, (570, 400), (608, 400))

    guardrail = (150, 500, 830, 590)
    rrect(d, guardrail, 8, fill=WHITE, outline=RED, width=2)
    _dashed_rect_outline(d, guardrail, RED, 2, dash_len=6, gap_len=4)
    ctext(d, (170, 518), "Guardrail (x3, one nudge each, then accept)", F_BOX_TITLE, RED, anchor="lm")
    wrap_and_draw_left(
        d, (170, 540),
        "not searched yet -> force search  \u00b7  searched but ungrounded -> admit the gap  \u00b7  "
        "document asked for but no artifact -> force create_artifact",
        F_BOX_MONO, INK_SOFT, 640, 17,
    )
    elbow(d, (500, 470), (500, 500), via="v")

    reply = (300, 630, 700, 700)
    note_box(
        d, reply, "Chat reply",
        ["+ citations (episode, guest, timestamp)", "+ artifact (if any), persisted"],
        INK, fill=WHITE,
    )
    elbow(d, (500, 590), (500, 630), via="v")

    ctext(
        d, (500, 740),
        "measured live: retrieval ~0.1s \u00b7 llama3.2:3b inference is the cost (~20-160s, CPU-only)",
        F_SMALL_MONO, INK_MUTED, anchor="mm",
    )

    sandbox_note = (75, 780, 830, 860)
    note_box(
        d, sandbox_note, "Artifact safety (2 layers)",
        [
            "server: allowlist strip (script/iframe/form), CSS @import + remote url() blocked",
            "client: <iframe sandbox=\"allow-scripts\"> no allow-same-origin + CSP default-src 'none'",
        ],
        AMBER, fill=AMBER_SOFT, border=AMBER,
    )

    # ================= RIGHT: BENCH / EVAL HARNESS (blue) =================
    right = (895, 155, 1690, 878)
    rrect(d, right, 14, fill=BLUE_SOFT, outline=BLUE, width=3)
    ctext(d, (915, 175), "BENCH \u2014 Eval Harness", F_SECTION, BLUE, anchor="lm")
    ctext(d, (915, 198), "app/evals/  \u00b7  retrieval only, no model \u2014 seconds, not minutes", F_SECTION_SUB, INK_MUTED, anchor="lm")

    golden = (915, 225, 1670, 320)
    note_box(
        d, golden, "Golden Scenario Set",
        [
            "24 labeled questions: 14 in-domain",
            "(7 named-guest, 7 broad topic)",
            "10 out-of-domain (unrelated to product/growth)",
        ],
        BLUE,
    )

    runner = (915, 345, 1670, 415)
    note_box(
        d, runner, "Scenario Runner",
        ["calls rag.retriever.search() directly against", "the real corpus \u2014 no agent loop, no live model"],
        BLUE,
    )
    arrow(d, (1292, 320), (1292, 343))

    scorers = (915, 440, 1670, 545)
    note_box(
        d, scorers, "Scorers",
        [
            "Grounded Answer Rate  (target >= 80%)",
            "False-Ground Rate     (must be 0%)",
            "guest-match precision \u00b7 latency p50/p95",
        ],
        BLUE,
    )
    arrow(d, (1292, 415), (1292, 438))

    trace = (915, 570, 1280, 640)
    note_box(d, trace, "Trace", ["every question: query, sim,", "grounded, latency \u2014 printed"], INK)

    gate_c = (1470, 605)
    gr = 55
    pts = [
        (gate_c[0], gate_c[1] - gr), (gate_c[0] + gr, gate_c[1]),
        (gate_c[0], gate_c[1] + gr), (gate_c[0] - gr, gate_c[1]),
    ]
    d.polygon(pts, outline=BLUE, width=3, fill=WHITE)
    ctext(d, (gate_c[0], gate_c[1] - 8), "Gate", F_BOX_TITLE, BLUE, anchor="mm")
    ctext(d, (gate_c[0], gate_c[1] + 12), "score >= floor?", F_BOX_MONO, INK_SOFT, anchor="mm")
    arrow(d, (1280, 605), (1415, 605))

    release = (915, 690, 1290, 780)
    note_box(
        d, release, "Release (pass)",
        ["commit RETRIEVAL_MIN_SIMILARITY", "with the measured data in the", "config.py comment"],
        GREEN, fill=GREEN_SOFT, border=GREEN,
    )
    diagnose = (1330, 690, 1670, 780)
    rrect(d, diagnose, 8, fill=WHITE, outline=RED, width=2)
    _dashed_rect_outline(d, diagnose, RED, 2, dash_len=6, gap_len=4)
    ctext(d, (1350, 708), "Diagnose (below floor)", F_BOX_TITLE, RED, anchor="lm")
    wrap_and_draw_left(
        d, (1350, 730),
        "print real similarity scores -> find the actual separation between in-domain and out-of-domain",
        F_BOX_MONO, INK_SOFT, 300, 16,
    )

    # Gate -> Diagnose (below floor): straight down, gate sits above diagnose's x-range.
    arrow(d, (1470, 660), (1470, 688), color=RED)
    ctext(d, (1500, 675), "below floor", F_SMALL_MONO, RED, anchor="lm")
    # Gate -> Release (pass): elbow left then down.
    elbow(d, (1415, 605), (1100, 688), color=GREEN, via="h")
    ctext(d, (1150, 592), "pass", F_SMALL_MONO, GREEN, anchor="lm")

    ctext(
        d, (1292, 835),
        "first run: 100% grounded, 80% false-ground (FAIL)  \u2192  after fix: 100% / 0% (PASS)",
        F_SMALL_MONO, INK_MUTED, anchor="mm",
    )

    # cross-link between the two halves
    elbow(d, (830, 400), (895, 400), via="h", color=INK_MUTED)
    ctext(d, (862, 378), "shares", F_SMALL_MONO, INK_MUTED, anchor="mm")
    ctext(d, (862, 424), "retriever()", F_SMALL_MONO, INK_MUTED, anchor="mm")

    # ---- Bottom row: Postgres / Ingestion / Ollama+Cloud (memory-layer style) ----
    mem_y0 = 925
    mem_h = 175
    ctext(d, (55, mem_y0 - 12), "DATA LAYER", F_SECTION_SUB, INK_MUTED, anchor="lm")

    pg = (55, mem_y0, 590, mem_y0 + mem_h)
    note_box(
        d, pg, "Postgres + pgvector",
        [
            "episodes \u00b7 chunks (HNSW + tsvector)",
            "sessions / messages \u00b7 artifacts",
            "ingestion_runs (audit trail)",
        ],
        INK,
    )

    ing = (615, mem_y0, 1150, mem_y0 + mem_h)
    note_box(
        d, ing, "Ingestion pipeline",
        [
            "clone -> chunk (speaker turns) -> flag",
            "sponsors -> embed (fastembed/ONNX) -> upsert",
            "303 episodes, idempotent + resumable",
        ],
        INK,
    )

    models = (1175, mem_y0, 1690, mem_y0 + mem_h)
    note_box(
        d, models, "Models  (LLM_PROVIDER=...)",
        [
            "ollama: llama3.2:3b, local, no API key",
            "openai_compat: HF router / OpenAI / Groq",
            "swap with one env var, no code change",
        ],
        INK,
    )

    # ---- Stack row ----
    stack_y0 = mem_y0 + mem_h + 30
    ctext(d, (55, stack_y0 - 6), "STACK", F_SECTION_SUB, INK_MUTED, anchor="lm")
    stack_items = [
        ("FastAPI", "backend + agent loop"),
        ("Next.js", "chat + artifact viewer"),
        ("Postgres + pgvector", "hybrid retrieval"),
        ("Ollama", "local, no key required"),
        ("OpenAI-compatible", "cloud, one env var"),
        ("fastembed / ONNX", "CPU-only embeddings"),
    ]
    n = len(stack_items)
    gap = 16
    pw = (W - 60 - gap * (n - 1)) / n
    x = 30
    py0 = stack_y0 + 14
    py1 = py0 + 56
    for title, sub in stack_items:
        box = (x, py0, x + pw, py1)
        rrect(d, box, 20, fill=WHITE, outline=GREY_BORDER, width=2)
        cx = x + pw / 2
        ctext(d, (cx, py0 + 18), title, F_PILL, INK, anchor="mm")
        ctext(d, (cx, py0 + 38), sub, F_PILL_SUB, INK_MUTED, anchor="mm")
        x += pw + gap

    # ---- What this fixes ----
    fix_y0 = py1 + 26
    ctext(d, (55, fix_y0 - 6), "WHAT THIS FIXES", F_SECTION_SUB, INK_MUTED, anchor="lm")
    fix_y = fix_y0 + 22
    fix_y = wrap_and_draw_left(
        d, (55, fix_y),
        "A RAG assistant that only demos well on the questions you happened to try isn't trustworthy on the "
        "ones you didn't. The eval harness above turns \u201cit seems grounded\u201d into a measured number \u2014 and it "
        "caught a real defect on its first run: an eyeballed similarity floor (0.55) let 80% of out-of-domain "
        "questions ground as fact. Fixed by measuring the actual score gap (0.71\u20130.81 in-domain vs. "
        "0.54\u20130.66 out-of-domain) and moving the floor into it (0.69), instead of guessing again.",
        F_BODY, INK_SOFT, W - 110, 23,
    )
    wrap_and_draw_left(
        d, (55, fix_y + 6),
        "Built on: hybrid retrieval (pgvector cosine + Postgres full-text, fused with RRF), a provider-agnostic "
        "LLM interface (Ollama today, any OpenAI-compatible cloud with one env var), and three deterministic "
        "guards that turn \u201cthe prompt says to search first\u201d from a probability into a guarantee on a 3B local model.",
        F_BODY, INK_SOFT, W - 110, 23,
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "docs", "architecture_v2.png")
    img.save(out)
    print(f"saved {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
