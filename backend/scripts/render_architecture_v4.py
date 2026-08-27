"""One-off script that renders docs/architecture_v4.png.

Structure inspired by the user-provided HARNESS + BENCH reference diagram,
adapted to what this project actually has: one live agent harness (the real
chat-turn loop) and BENCH split into the two eval harnesses that actually
exist -- an LLM/retrieval harness (run_eval.py) and an agent-scenario harness
(run_agent_eval.py). Text kept to box titles and one short line each -- no
embedded paragraphs.

Run with: python render_architecture_v4.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1820, 1060

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
PURPLE = (106, 55, 145)
PURPLE_SOFT = (243, 236, 250)
GREY_BORDER = (150, 150, 145)
WHITE = (255, 255, 255)

FONT_DIR = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + name, size)


F_TITLE = font("comicbd.ttf", 36)
F_SUBTITLE = font("comic.ttf", 17)
F_HDR_TAG = font("consolab.ttf", 13)
F_META = font("consola.ttf", 12)
F_SECTION = font("comicbd.ttf", 18)
F_SECTION_SUB = font("consola.ttf", 12)
F_BOX_TITLE = font("comicbd.ttf", 15)
F_BOX_SUB = font("consola.ttf", 11)
F_CAPTION = font("consola.ttf", 12)


def rrect(d, box_, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box_, radius=r, fill=fill, outline=outline, width=width)


def dashed_rect(d, box_, color, width=2, dash_len=6, gap_len=4):
    x0, y0, x1, y1 = box_
    for p1, p2 in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
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
            d.line([(x1 + dx * pos, y1 + dy * pos), (x1 + dx * end, y1 + dy * end)], fill=color, width=width)
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
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    arrowhead(d, p2, angle, color)


def elbow(d, p1, p2, color=INK_MUTED, width=2, via="h"):
    mid = (p2[0], p1[1]) if via == "h" else (p1[0], p2[1])
    d.line([p1, mid], fill=color, width=width)
    d.line([mid, p2], fill=color, width=width)
    angle = math.atan2(p2[1] - mid[1], p2[0] - mid[0])
    arrowhead(d, p2, angle, color)


def box(d, rect, title, sub=None, fill=WHITE, border=GREY_BORDER, title_color=INK):
    rrect(d, rect, 8, fill=fill, outline=border, width=2)
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ctext(d, (cx, cy - 8), title, F_BOX_TITLE, title_color)
        ctext(d, (cx, cy + 10), sub, F_BOX_SUB, INK_MUTED)
    else:
        ctext(d, (cx, cy), title, F_BOX_TITLE, title_color)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # ---- Header ----
    ctext(d, (36, 38), "The Lenny Growth Assistant", F_TITLE, INK, anchor="lm")
    ctext(
        d, (36, 70),
        "A grounded RAG agent, gated by two harnesses before it ships: one for the LLM, one for the agent.",
        F_SUBTITLE, INK_SOFT, anchor="lm",
    )
    ctext(d, (W - 36, 26), "LENNY GROWTH ASSISTANT \u2014 V2 ARCHITECTURE", F_HDR_TAG, RED, anchor="rm")
    ctext(d, (W - 36, 44), "ADR-0002 \u00b7 status: shipped \u00b7 2026-08-28", F_META, INK_MUTED, anchor="rm")

    left_bottom = 858  # fixed: matches the known height of the left panel's content below

    # ================= LEFT: HARNESS (live agent loop) =================
    left = (48, 122, 900, left_bottom)
    rrect(d, left, 14, fill=RED_SOFT, outline=RED, width=3)
    ltext(d, (68, 145), "HARNESS \u2014 Agent Loop", F_SECTION, RED)
    ltext(d, (68, 166), "run_agent()  \u00b7  one call per chat turn", F_SECTION_SUB, INK_MUTED)

    browser = (68, 195, 320, 250)
    box(d, browser, "Browser", "chat + artifacts")
    nextjs = (350, 195, 590, 250)
    box(d, nextjs, "Next.js UI", "sessions \u00b7 citations", fill=BLUE_SOFT)
    arrow(d, (320, 222), (348, 222))
    api = (620, 195, 870, 250)
    box(d, api, "FastAPI /chat", "persists, then runs agent", fill=WHITE, border=RED)
    arrow(d, (590, 222), (618, 222))

    router_c = (478, 350)
    rr = 62
    d.ellipse([router_c[0] - rr, router_c[1] - rr, router_c[0] + rr, router_c[1] + rr],
              outline=AMBER, width=3, fill=AMBER_SOFT)
    ctext(d, (router_c[0], router_c[1] - 10), "Router / Agent", F_BOX_TITLE, AMBER)
    ctext(d, (router_c[0], router_c[1] + 12), "5 guards", F_BOX_SUB, INK_SOFT)
    elbow(d, (745, 250), (478, 288), via="v")

    tools = (620, 320, 870, 400)
    box(d, tools, "Tool registry", "search \u00b7 essay \u00b7 artifact", fill=GREEN_SOFT, border=GREEN)
    arrow(d, (540, 350), (618, 350))

    guardrail = (150, 460, 830, 520)
    dashed_rect(d, guardrail, RED, 2, 6, 4)
    ctext(d, (490, 480), "force search \u00b7 block ungrounded artifact \u00b7 block redundant artifact",
          F_BOX_SUB, RED)
    ctext(d, (490, 500), "force artifact when asked \u00b7 no tools on trivial messages", F_BOX_SUB, RED)
    elbow(d, (478, 412), (478, 460), via="v")

    reply = (300, 560, 700, 620)
    box(d, reply, "Chat reply", "+ citations \u00b7 + artifact")
    elbow(d, (478, 520), (478, 560), via="v")

    sandbox = (75, 660, 870, 730)
    box(d, sandbox, "Artifact sandbox", "server sanitiser + iframe sandbox, 2 layers",
        fill=AMBER_SOFT, border=AMBER)

    models_y = 770
    ollama = (150, models_y, 460, models_y + 68)
    box(d, ollama, "Ollama", "local \u00b7 no API key", fill=GREEN_SOFT, border=GREEN)
    cloud = (500, models_y, 810, models_y + 68)
    box(d, cloud, "OpenAI-compatible", "cloud \u00b7 one env var")
    elbow(d, (478, 730), (305, models_y), via="v")
    elbow(d, (478, 730), (655, models_y), via="v")

    # ================= RIGHT: BENCH (two eval harnesses) =================
    right_x0, right_x1 = 935, W - 48
    col_w = (right_x1 - right_x0 - 3 * 20) / 2
    lx0 = right_x0 + 20
    rx0 = lx0 + col_w + 20

    def harness_column(x0, title, subtitle, accent, accent_soft, steps, caption):
        x1 = x0 + col_w
        y = 260
        step_h = 74
        gap = 14
        n = len(steps)
        col_bottom = y + n * step_h + (n - 1) * gap + 54
        col = (x0, 195, x1, col_bottom)
        rrect(d, col, 12, fill=accent_soft, outline=accent, width=2)
        ctext(d, ((x0 + x1) / 2, 218), title, F_SECTION, accent)
        ctext(d, ((x0 + x1) / 2, 238), subtitle, F_SECTION_SUB, INK_MUTED)
        for i, (step_title, step_sub) in enumerate(steps):
            b = (x0 + 16, y, x1 - 16, y + step_h)
            box(d, b, step_title, step_sub, fill=WHITE, border=accent)
            if i < n - 1:
                arrow(d, ((x0 + x1) / 2, y + step_h), ((x0 + x1) / 2, y + step_h + gap), color=accent)
            y += step_h + gap
        ctext(d, ((x0 + x1) / 2, y + 12), caption, F_CAPTION, INK_SOFT)
        return col_bottom

    col_bottom_llm = harness_column(
        lx0, "LLM HARNESS", "run_eval.py \u2014 retrieval only",
        BLUE, BLUE_SOFT,
        [
            ("Golden questions", "24: 14 in-domain, 10 out-of-domain"),
            ("rag.retriever.search()", "real corpus, no model"),
            ("Scorer", "Grounded Rate \u2265 80%  \u00b7  False-Ground = 0%"),
            ("Gate", "pass -> commit threshold / fail -> diagnose"),
        ],
        "found: 0.55 floor let 80% of\nout-of-domain Qs ground \u2192 fixed to 0.69",
    )
    col_bottom_agent = harness_column(
        rx0, "AGENT HARNESS", "run_agent_eval.py \u2014 real model",
        PURPLE, PURPLE_SOFT,
        [
            ("Golden scenarios", "8 conversations, real run_agent()"),
            ("Real model + guards", "llama3.2:3b \u00b7 real tool registry"),
            ("Scorer", "tool-call \u00b7 refusal \u00b7 redundant-call checks"),
            ("Gate", "pass -> ship / fail -> diagnose + fix"),
        ],
        "found: 2 live hallucinations \u2014\ncreate_artifact bypassed search entirely",
    )

    right_bottom = max(col_bottom_llm, col_bottom_agent) + 34
    right = (right_x0, 122, right_x1, right_bottom)
    rrect(d, right, 14, fill=None, outline=(90, 90, 100), width=3)
    ltext(d, (right_x0 + 20, 145), "BENCH \u2014 Two Harnesses, One Gate", F_SECTION, INK)
    ltext(d, (right_x0 + 20, 166), "app/evals/  \u00b7  no live model in the LLM harness", F_SECTION_SUB, INK_MUTED)

    arrow(d, (900, 350), (right_x0 - 2, 350), color=INK_MUTED)

    outer_bottom = max(left_bottom, right_bottom) + 20
    dashed_rect(d, (24, 100, W - 24, outer_bottom), GREY_BORDER, 2, 6, 5)

    # ---- Bottom: data layer ----
    dl_y = outer_bottom + 30
    ltext(d, (48, dl_y - 8), "DATA LAYER", F_SECTION_SUB, INK_MUTED)
    dl_h = 80
    dl_y0 = dl_y + 10
    pg = (48, dl_y0, 620, dl_y0 + dl_h)
    box(d, pg, "Postgres + pgvector", "episodes \u00b7 chunks \u00b7 sessions \u00b7 artifacts")
    ing = (650, dl_y0, 1220, dl_y0 + dl_h)
    box(d, ing, "Ingestion", "clone \u2192 chunk \u2192 embed \u2192 upsert, 303 episodes")
    models = (1250, dl_y0, W - 48, dl_y0 + dl_h)
    box(d, models, "Model config", "LLM_PROVIDER=ollama | openai_compat")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "docs", "architecture_v4.png")
    img.save(out)
    print(f"saved {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
