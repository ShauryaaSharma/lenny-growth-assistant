"""One-off script that renders docs/architecture_v3.png.

A stripped-down version of v2: just the diagram, a heading, and one line of
explanation. No bullet lists inside boxes, no guardrail enumeration, no
closing narrative -- box titles and arrows only.

Run with: python render_architecture_v3.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 745

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
GREY_BORDER = (150, 150, 145)
WHITE = (255, 255, 255)

FONT_DIR = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + name, size)


F_TITLE = font("comicbd.ttf", 38)
F_SUBTITLE = font("comic.ttf", 18)
F_HDR_TAG = font("consolab.ttf", 13)
F_META = font("consola.ttf", 13)
F_BOX_TITLE = font("comicbd.ttf", 18)
F_BOX_SUB = font("consola.ttf", 13)
F_SECTION = font("comicbd.ttf", 16)


def rrect(d, box, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def dashed_rect(d, box, color, width=2, dash_len=7, gap_len=5):
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for p1, p2 in edges:
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


def box(d, rect, title, sub=None, fill=WHITE, border=GREY_BORDER, title_color=INK):
    rrect(d, rect, 10, fill=fill, outline=border, width=2)
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ctext(d, (cx, cy - 9), title, F_BOX_TITLE, title_color)
        ctext(d, (cx, cy + 13), sub, F_BOX_SUB, INK_MUTED)
    else:
        ctext(d, (cx, cy), title, F_BOX_TITLE, title_color)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # ---- Header: heading + one line ----
    ctext(d, (40, 42), "The Lenny Growth Assistant", F_TITLE, INK, anchor="lm")
    ctext(
        d, (40, 78),
        "Grounded RAG over Lenny's Podcast, with deterministic agent guards and its own eval harness.",
        F_SUBTITLE, INK_SOFT, anchor="lm",
    )
    ctext(d, (W - 40, 28), "LENNY GROWTH ASSISTANT — ARCHITECTURE", F_HDR_TAG, RED, anchor="rm")
    ctext(d, (W - 40, 46), "2026-08-28", F_META, INK_MUTED, anchor="rm")

    top = 130

    # ---- Client -> API ----
    browser = (40, top, 250, top + 70)
    box(d, browser, "Browser", "chat + artifacts")

    nextjs = (300, top, 510, top + 70)
    box(d, nextjs, "Next.js UI", fill=BLUE_SOFT)
    arrow(d, (250, top + 35), (298, top + 35))

    # ---- FastAPI container ----
    api = (560, top - 25, 1460, top + 330)
    rrect(d, api, 16, fill=WHITE, outline=RED, width=3)
    ctext(d, (580, top - 4), "FastAPI \u2014 Agent Loop", F_SECTION, RED, anchor="lm")
    arrow(d, (510, top + 35), (558, top + 35))

    agent = (600, top + 25, 830, top + 105)
    box(d, agent, "Router / Agent", "5 deterministic guards", fill=RED_SOFT, border=RED)

    tools = (870, top + 25, 1130, top + 105)
    box(d, tools, "Tool registry", "search \u00b7 essay \u00b7 artifact", fill=GREEN_SOFT, border=GREEN)
    arrow(d, (830, top + 65), (868, top + 65))

    llm = (600, top + 145, 830, top + 225)
    box(d, llm, "LLMProvider", "chat() \u00b7 health()")
    arrow(d, (715, top + 105), (715, top + 143))

    sandbox = (870, top + 145, 1130, top + 225)
    box(d, sandbox, "Sanitiser + sandbox", "artifact safety, 2 layers", fill=AMBER_SOFT, border=AMBER)
    arrow(d, (1000, top + 105), (1000, top + 143))

    reply = (600, top + 265, 1130, top + 325)
    box(d, reply, "Chat reply", "+ citations \u00b7 + artifact", fill=WHITE)
    arrow(d, (715, top + 225), (715, top + 263))
    arrow(d, (1000, top + 225), (1000, top + 263))

    # ---- Models, below the API box ----
    model_y = top + 380
    ollama = (600, model_y, 830, model_y + 70)
    box(d, ollama, "Ollama", "local \u00b7 no API key", fill=GREEN_SOFT, border=GREEN)
    cloud = (900, model_y, 1130, model_y + 70)
    box(d, cloud, "OpenAI-compatible", "cloud \u00b7 one env var")
    elbow(d, (715, top + 330), (715, model_y), via="v")
    elbow(d, (715, top + 330), (1015, model_y), via="v")
    ctext(d, (865, model_y - 12), "LLM_PROVIDER = ...", F_META, INK_MUTED, anchor="mm")

    # ---- Postgres, left column ----
    pg = (40, top + 145, 250, top + 305)
    box(d, pg, "Postgres", "+ pgvector", fill=WHITE)
    elbow(d, (600, top + 65), (250, top + 200), via="h", color=INK_MUTED)

    # ---- BENCH: eval harness, right column ----
    bench = (1200, top + 145, 1460, top + 380)
    rrect(d, bench, 14, fill=BLUE_SOFT, outline=BLUE, width=3)
    ctext(d, (1330, top + 175), "BENCH", F_SECTION, BLUE, anchor="mm")
    ctext(d, (1330, top + 200), "eval harness", F_BOX_SUB, INK_MUTED, anchor="mm")
    box(d, (1220, top + 225, 1440, top + 275), "Retrieval eval", fill=WHITE)
    box(d, (1220, top + 290, 1440, top + 340), "Agent-scenario eval", fill=WHITE)
    elbow(d, (1130, top + 65), (1330, top + 145), via="h", color=BLUE)
    ctext(d, (1330, top + 130), "drives", F_META, BLUE, anchor="mm")

    # ---- Ingestion pipeline, bottom strip ----
    ing_y = model_y + 130
    ing = (40, ing_y, 1460, ing_y + 90)
    rrect(d, ing, 14, fill=WHITE, outline=GREY_BORDER, width=2)
    ctext(d, (60, ing_y + 22), "Ingestion", F_SECTION, INK, anchor="lm")
    steps = ["Clone corpus", "Chunk", "Embed", "Upsert \u2192 Postgres"]
    n = len(steps)
    sx0, sx1 = 260, 1440
    step_w = (sx1 - sx0 - 20 * (n - 1)) / n
    x = sx0
    for i, s in enumerate(steps):
        b = (x, ing_y + 15, x + step_w, ing_y + 70)
        rrect(d, b, 10, fill=BLUE_SOFT, outline=GREY_BORDER, width=2)
        ctext(d, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), s, F_BOX_SUB, INK)
        if i < n - 1:
            arrow(d, (x + step_w, ing_y + 42), (x + step_w + 20, ing_y + 42))
        x += step_w + 20

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "docs", "architecture_v3.png")
    img.save(out)
    print(f"saved {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
