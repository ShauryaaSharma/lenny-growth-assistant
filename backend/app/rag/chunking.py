"""Transcript parsing and chunking.

The corpus is one markdown file per episode: YAML frontmatter, then the body as
`Speaker (HH:MM:SS):` blocks.

Two decisions worth calling out:

1. **Chunk on speaker turns, not fixed character windows.** A podcast answer is
   a coherent unit of thought; splitting mid-sentence produces chunks that
   retrieve well but read badly when quoted back as evidence. We accumulate
   whole turns up to a token budget and overlap by whole turns.

2. **Sponsor reads are detected and flagged.** A meaningful slice of the corpus
   is ad copy ("This episode is brought to you by Linear..."). Left in, it
   dominates retrieval for commercial queries like "how should I price my
   product" and produces confidently-cited nonsense. We keep the rows for
   auditability but exclude them from retrieval.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

TURN_RE = re.compile(r"^(?P<speaker>[^\n(]{1,120}?)\s*\((?P<ts>\d{1,2}:\d{2}:\d{2})\):\s*$")

SPONSOR_MARKERS = (
    "brought to you by",
    "this episode is sponsored",
    "sponsored by",
    "thank you to our sponsor",
)
# Secondary markers only promote a turn to "sponsor" when it directly follows one.
SPONSOR_CONTINUATION_MARKERS = (
    "to sign up",
    "get 25% off",
    "free trial",
    "use code",
    "visit ",
    "head over to",
    "/lenny",
)


@dataclass
class EpisodeMeta:
    video_id: str
    guest: str
    title: str
    youtube_url: str
    publish_date: date | None
    duration_seconds: float | None
    description: str | None
    keywords: list[str]
    source_path: str
    content_hash: str


@dataclass
class Turn:
    speaker: str
    start_seconds: int
    text: str
    is_sponsor: bool = False


@dataclass
class ChunkData:
    ordinal: int
    speaker: str | None
    start_seconds: int | None
    end_seconds: int | None
    text: str
    token_count: int
    is_sponsor: bool = False
    turns: list[Turn] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens/word).

    Deliberately not a real tokenizer: chunk sizing only needs to be roughly
    right, and pulling in a tokenizer for it would add a heavy dependency that
    must match whichever model is currently selected -- which changes at runtime.
    """
    return max(1, int(len(text.split()) * 1.3))


def _ts_to_seconds(ts: str) -> int:
    h, m, s = (int(p) for p in ts.split(":"))
    return h * 3600 + m * 60 + s


def parse_frontmatter(raw: str, source_path: str) -> tuple[EpisodeMeta, str]:
    """Split `---` frontmatter from the body. Raises ValueError on malformed input."""
    if not raw.startswith("---"):
        raise ValueError(f"{source_path}: missing YAML frontmatter")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{source_path}: unterminated YAML frontmatter")

    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2]

    # A handful of episodes in the upstream corpus ship with empty metadata
    # (video_id: ''). They still contain a full transcript, so we synthesise a
    # stable id from the folder slug rather than discarding real content. Such
    # episodes cite by title only -- there is no timestamp to deep-link to.
    video_id = str(meta.get("video_id") or "").strip()
    if not video_id:
        video_id = f"slug:{Path(source_path).parent.name}"

    youtube_url = str(meta.get("youtube_url") or "").strip()
    if not youtube_url and not video_id.startswith("slug:"):
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    pub = meta.get("publish_date")
    if isinstance(pub, datetime):
        pub = pub.date()
    elif isinstance(pub, str):
        try:
            pub = datetime.strptime(pub.strip(), "%Y-%m-%d").date()
        except ValueError:
            pub = None
    elif not isinstance(pub, date):
        pub = None

    kw = meta.get("keywords") or []
    if isinstance(kw, str):
        kw = [k.strip() for k in kw.split(",") if k.strip()]

    duration = meta.get("duration_seconds")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None

    description = meta.get("description")
    return (
        EpisodeMeta(
            video_id=video_id,
            guest=str(meta.get("guest") or "Unknown guest").strip(),
            title=str(meta.get("title") or "Untitled episode").strip(),
            youtube_url=youtube_url,
            publish_date=pub,
            duration_seconds=duration,
            description=str(description).strip() if description else None,
            keywords=[str(k) for k in kw],
            source_path=source_path,
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        ),
        body,
    )


def parse_turns(body: str) -> list[Turn]:
    """Extract `Speaker (HH:MM:SS):` blocks in document order."""
    turns: list[Turn] = []
    current: Turn | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            text = "\n".join(buffer).strip()
            if text:
                current.text = text
                turns.append(current)

    for line in body.splitlines():
        match = TURN_RE.match(line.strip())
        if match:
            flush()
            buffer = []
            current = Turn(
                speaker=match.group("speaker").strip(),
                start_seconds=_ts_to_seconds(match.group("ts")),
                text="",
            )
        elif current is not None:
            buffer.append(line)
    flush()

    _flag_sponsors(turns)
    return turns


def _flag_sponsors(turns: list[Turn]) -> None:
    """Mark ad reads.

    A turn containing a primary marker opens a sponsor block; adjacent CTA turns
    by the same speaker are swept in until a turn looks like ordinary talk again.
    """
    in_block = False
    block_speaker: str | None = None
    for turn in turns:
        low = turn.text.lower()
        if any(m in low for m in SPONSOR_MARKERS):
            turn.is_sponsor = True
            in_block = True
            block_speaker = turn.speaker
            continue
        if in_block and turn.speaker == block_speaker:
            if any(m in low for m in SPONSOR_CONTINUATION_MARKERS):
                turn.is_sponsor = True
                continue
        in_block = False
        block_speaker = None


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long_turn(turn: Turn, target_tokens: int) -> list[Turn]:
    """Break a monologue longer than the chunk budget on sentence boundaries.

    Guests occasionally talk for several thousand tokens without interruption.
    Left whole, such a turn produces a chunk that overflows the embedding
    model's context window and is silently truncated at encode time -- the tail
    becomes unretrievable. Splitting keeps every sentence reachable.
    """
    if estimate_tokens(turn.text) <= target_tokens:
        return [turn]

    parts: list[Turn] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for sentence in _SENTENCE_RE.split(turn.text):
        tokens = estimate_tokens(sentence)
        if buffer and buffer_tokens + tokens > target_tokens:
            parts.append(
                Turn(turn.speaker, turn.start_seconds, " ".join(buffer), turn.is_sponsor)
            )
            buffer, buffer_tokens = [], 0
        buffer.append(sentence)
        buffer_tokens += tokens
    if buffer:
        parts.append(Turn(turn.speaker, turn.start_seconds, " ".join(buffer), turn.is_sponsor))
    return parts


def chunk_turns(
    turns: list[Turn], target_tokens: int = 400, overlap_tokens: int = 80
) -> list[ChunkData]:
    """Pack whole turns into ~target_tokens chunks with a trailing-turn overlap.

    Sponsor turns are chunked separately from content turns so a single chunk is
    never half advertisement and half substance.
    """
    chunks: list[ChunkData] = []
    ordinal = 0
    buffer: list[Turn] = []
    buffer_tokens = 0

    def emit() -> list[Turn]:
        """Write the buffered turns out as a chunk; return the overlap carry."""
        nonlocal ordinal, buffer_tokens
        if not buffer:
            return []
        text = "\n\n".join(f"{t.speaker}: {t.text}" for t in buffer)
        chunks.append(
            ChunkData(
                ordinal=ordinal,
                speaker=buffer[0].speaker,
                start_seconds=buffer[0].start_seconds,
                end_seconds=buffer[-1].start_seconds,
                text=text,
                token_count=estimate_tokens(text),
                is_sponsor=buffer[0].is_sponsor,
                turns=list(buffer),
            )
        )
        ordinal += 1

        # Carry the tail of this chunk into the next so a thought that straddles
        # the boundary is retrievable from either side.
        carry: list[Turn] = []
        carried = 0
        for turn in reversed(buffer):
            t = estimate_tokens(turn.text)
            if carried + t > overlap_tokens:
                break
            carry.insert(0, turn)
            carried += t
        buffer_tokens = carried
        return carry

    expanded: list[Turn] = []
    for turn in turns:
        expanded.extend(_split_long_turn(turn, target_tokens))

    for turn in expanded:
        # A sponsor/content transition forces a boundary.
        if buffer and buffer[0].is_sponsor != turn.is_sponsor:
            emit()
            buffer, buffer_tokens = [], 0

        tokens = estimate_tokens(turn.text)
        if buffer and buffer_tokens + tokens > target_tokens:
            buffer = emit()
        buffer.append(turn)
        buffer_tokens += tokens

    emit()
    return chunks


def parse_transcript_file(
    path: Path, target_tokens: int = 400, overlap_tokens: int = 80
) -> tuple[EpisodeMeta, list[ChunkData]]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw, str(path))
    turns = parse_turns(body)
    return meta, chunk_turns(turns, target_tokens, overlap_tokens)
