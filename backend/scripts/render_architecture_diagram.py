"""One-off script that renders docs/architecture.png.

Not part of the running application -- generates a static architecture
diagram for the README. Run with: python render_architecture_diagram.py
(from backend/scripts/, or any cwd -- output path is repo-root-relative).
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 1000

BG = (250, 250, 252)
INK = (17, 24, 39)
INK_SOFT = (71, 85, 105)
INK_MUTED = (100, 116, 139)
BOX_BORDER = (203, 213, 225)
BOX_FILL = (255, 255, 255)
ACCENT = (79, 70, 229)
ACCENT_SOFT = (238, 242, 255)
GREEN = (5, 150, 105)
GREEN_SOFT = (236, 253, 245)
AMBER = (180, 83, 9)
AMBER_SOFT = (255, 251, 235)
ARROW = (100, 116, 139)

FONT_DIR = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + name, size)


F_TITLE = font("segoeuib.ttf", 34)
F_SUB = font("segoeui.ttf", 20)
F_BOX_TITLE = font("segoeuib.ttf", 20)
F_BOX_SUB = font("segoeui.ttf", 15)
F_LABEL = font("segoeui.ttf", 15)
F_SMALL = font("segoeui.ttf", 13)
F_MONO = font("consola.ttf", 14)


def rrect(draw, box, radius, fill, outline, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def ctext(draw, xy, text, fnt, fill, anchor="mm"):
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def arrowhead(draw, tip, angle, color, size=9):
    left = (tip[0] - size * math.cos(angle - 0.45), tip[1] - size * math.sin(angle - 0.45))
    right = (tip[0] - size * math.cos(angle + 0.45), tip[1] - size * math.sin(angle + 0.45))
    draw.polygon([tip, left, right], fill=color)


def straight_arrow(draw, p1, p2, color=ARROW, width=3):
    draw.line([p1, p2], fill=color, width=width)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    arrowhead(draw, p2, angle, color)


def elbow_arrow(draw, p1, p2, color=ARROW, width=3, via="h"):
    """Orthogonal connector: 'h' bends horizontally-then-vertically, 'v' the reverse."""
    if via == "h":
        mid = (p2[0], p1[1])
    else:
        mid = (p1[0], p2[1])
    draw.line([p1, mid], fill=color, width=width)
    draw.line([mid, p2], fill=color, width=width)
    angle = math.atan2(p2[1] - mid[1], p2[0] - mid[0])
    arrowhead(draw, p2, angle, color)


def title_box(draw, box, title, subtitle_lines=None, fill=BOX_FILL, radius=14):
    rrect(draw, box, radius, fill, BOX_BORDER)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    if subtitle_lines:
        ctext(draw, (cx, y0 + 26), title, F_BOX_TITLE, INK)
        y = y0 + 50
        for line, color in subtitle_lines:
            ctext(draw, (cx, y), line, F_BOX_SUB, color)
            y += 20
    else:
        ctext(draw, (cx, (y0 + y1) / 2), title, F_BOX_TITLE, INK)


def bullet_box(draw, box, lines, header, header_color=INK, fill=BOX_FILL):
    rrect(draw, box, 12, fill, BOX_BORDER)
    x0, y0, x1, y1 = box
    ctext(draw, ((x0 + x1) / 2, y0 + 22), header, F_BOX_TITLE, header_color)
    y = y0 + 44
    draw.line([(x0 + 14, y), (x1 - 14, y)], fill=BOX_BORDER, width=1)
    y += 14
    for line in lines:
        draw.text((x0 + 16, y), f"\u2022 {line}", font=F_LABEL, fill=INK_SOFT)
        y += 24


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    ctext(d, (W / 2, 38), "The Lenny Growth Assistant", F_TITLE, INK)
    ctext(d, (W / 2, 74), "Architecture", F_SUB, INK_MUTED)

    # ---- Browser ----
    browser = (50, 140, 280, 220)
    title_box(d, browser, "Browser", [("chat + artifact viewer", INK_SOFT)])

    # ---- Next.js UI ----
    nextjs = (340, 130, 620, 230)
    title_box(
        d, nextjs, "Next.js UI",
        [("sessions \u00b7 citations", INK_SOFT), ("provider badge", INK_SOFT)],
        fill=ACCENT_SOFT,
    )
    straight_arrow(d, (280, 180), (338, 180))

    # ---- FastAPI container ----
    api_box = (680, 110, 1450, 500)
    rrect(d, api_box, 18, (255, 255, 255), BOX_BORDER, width=2)
    ctext(d, (1065, 136), "FastAPI backend", F_BOX_TITLE, INK)
    straight_arrow(d, (620, 178), (678, 178))
    ctext(d, (649, 160), "REST", F_SMALL, INK_MUTED)

    agent_box = (712, 170, 970, 260)
    title_box(d, agent_box, "Agent loop", [("3 deterministic guards", INK_SOFT)], fill=ACCENT_SOFT)

    tools_box = (1010, 170, 1418, 320)
    bullet_box(
        d, tools_box,
        ["search_transcripts", "write_ship30_essay", "create_artifact"],
        header="Tool registry", header_color=GREEN, fill=GREEN_SOFT,
    )
    straight_arrow(d, (970, 215), (1008, 215))

    provider_box = (712, 300, 970, 390)
    title_box(d, provider_box, "LLMProvider", [("chat()  \u00b7  health()", INK_SOFT)])
    straight_arrow(d, (841, 260), (841, 298))

    sanitize_box = (1010, 350, 1418, 460)
    bullet_box(
        d, sanitize_box,
        ["allowlist strip: script/iframe/form", "CSS @import + remote url() stripped"],
        header="Sanitiser (server-side)", header_color=AMBER, fill=AMBER_SOFT,
    )
    straight_arrow(d, (1214, 320), (1214, 348))

    # ---- Postgres (left of API box, connected with an elbow) ----
    pg_box = (50, 300, 280, 500)
    bullet_box(
        d, pg_box,
        ["episodes", "chunks (pgvector +", "  tsvector, HNSW)", "sessions / messages", "artifacts", "ingestion_runs"],
        header="Postgres",
    )
    ctext(d, (165, 510), "SQLAlchemy (async)", F_SMALL, INK_MUTED)
    elbow_arrow(d, (712, 220), (280, 340), via="v")

    # ---- Ollama / Cloud (below the API box) ----
    ollama_box = (712, 600, 970, 700)
    title_box(
        d, ollama_box, "Ollama",
        [("local \u00b7 llama3.2:3b", INK_SOFT), ("no API key required", GREEN)],
        fill=GREEN_SOFT,
    )
    cloud_box = (1160, 600, 1418, 700)
    title_box(
        d, cloud_box, "OpenAI-compatible",
        [("HF router \u00b7 OpenAI \u00b7 Groq", INK_SOFT), ("one env var to switch", INK_MUTED)],
    )

    straight_arrow(d, (841, 390), (841, 598))
    elbow_arrow(d, (1214, 462), (1289, 598), via="v")
    ctext(d, (1065, 540), "LLM_PROVIDER = ollama | openai_compat", F_MONO, INK_MUTED)

    # ---- Ingestion pipeline strip ----
    ing_y0 = 760
    ing_box = (50, ing_y0, 1450, ing_y0 + 200)
    rrect(d, ing_box, 16, (255, 255, 255), BOX_BORDER)
    ctext(d, (750, ing_y0 + 26), "Ingestion pipeline  (backend/app/rag/ingest.py)", F_BOX_TITLE, INK)

    steps = [
        ("Clone corpus", ["git, shallow"]),
        ("Parse + chunk", ["speaker turns", "~400 tok, 80 overlap"]),
        ("Flag sponsors", ["kept, excluded", "from embedding"]),
        ("Embed", ["fastembed / ONNX", "bge-small-en-v1.5"]),
        ("Upsert", ["1 txn / episode", "content-hashed"]),
    ]
    n = len(steps)
    margin, gap = 90, 22
    step_w = (1400 - 2 * margin - gap * (n - 1)) / n
    x = 50 + margin
    y0, y1 = ing_y0 + 56, ing_y0 + 168
    step_centers = []
    for title, sublines in steps:
        box = (x, y0, x + step_w, y1)
        rrect(d, box, 12, ACCENT_SOFT, BOX_BORDER)
        cx = x + step_w / 2
        ctext(d, (cx, y0 + 22), title, F_BOX_SUB, INK)
        ly = y0 + 46
        for line in sublines:
            ctext(d, (cx, ly), line, F_SMALL, INK_MUTED)
            ly += 17
        step_centers.append((x, x + step_w))
        x += step_w + gap

    mid_y = (y0 + y1) / 2
    for i in range(len(step_centers) - 1):
        straight_arrow(d, (step_centers[i][1], mid_y), (step_centers[i + 1][0], mid_y))

    ctext(
        d, (750, ing_y0 + 182),
        "303 episodes \u2192 ~17,800 chunks \u00b7 idempotent, content-hashed, resumable",
        F_SMALL, INK_MUTED,
    )
    elbow_arrow(d, (165, 500), (165, ing_y0), via="v")

    ctext(d, (W / 2, H - 20), "Full detail: docs/architecture.md", F_SMALL, INK_MUTED)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "docs", "architecture.png")
    img.save(out)
    print(f"saved {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
