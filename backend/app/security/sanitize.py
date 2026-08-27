"""Artifact sanitisation -- layer 1 of 2.

Generated HTML is treated as untrusted input, because it is: it is produced by a
language model steered by user text, and the corpus it draws on is third-party
content. Prompt injection in either could otherwise become stored XSS.

The defence is two independent layers, either of which is sufficient on its own:

  **Layer 1 (here, server-side).** Parse and rewrite the HTML against a strict
  allowlist before it is ever persisted. Only the sanitised form is stored, so a
  later bug in the viewer cannot resurrect an unsafe payload from the database.

  **Layer 2 (client-side, see frontend ArtifactViewer).** Render inside an
  iframe with `sandbox="allow-scripts"` and *without* `allow-same-origin`, which
  puts the document on an opaque origin: it cannot read the parent DOM, our
  cookies, or localStorage. An injected CSP of `default-src 'none'` blocks all
  network egress, so even a successful script injection cannot exfiltrate.

What is permitted, and why:
  * Structural and text markup, tables, lists, headings -- the point of the feature.
  * `<style>` blocks and `style` attributes -- the brief asks for HTML/CSS
    artifacts, and CSS is where most of the value is. External references inside
    CSS (`@import`, `url(https://...)`) are stripped: they are a real
    exfiltration channel that leaks the viewer's IP and referrer.
  * `data:` images, so charts and diagrams work offline.

What is blocked:
  * `<script>`, `<iframe>`, `<object>`, `<embed>`, `<form>`, `<input>`, `<base>`,
    `<link>`, `<meta>` -- code execution, clickjacking, credential harvesting,
    and base-URL hijacking.
  * Every `on*` event-handler attribute.
  * `javascript:`, `vbscript:`, and `data:text/html` URLs.
"""

from __future__ import annotations

import re

import nh3

ALLOWED_TAGS: set[str] = {
    "a", "abbr", "article", "aside", "b", "blockquote", "br", "caption", "cite",
    "code", "col", "colgroup", "dd", "del", "details", "div", "dl", "dt", "em",
    "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "i", "img", "ins", "kbd", "li", "main", "mark", "nav", "ol", "p", "pre",
    "q", "s", "samp", "section", "small", "span", "strong", "sub", "summary",
    "sup", "style", "table", "tbody", "td", "tfoot", "th", "thead", "time", "tr",
    "u", "ul", "var", "wbr",
}

ALLOWED_ATTRIBUTES: dict[str, set[str]] = {
    "*": {"class", "id", "style", "title", "lang", "dir", "role", "aria-label"},
    # `rel` is deliberately absent: nh3's link_rel option sets it for us, and
    # allowing both is rejected outright.
    "a": {"href", "target"},
    "img": {"src", "alt", "width", "height", "loading"},
    "td": {"colspan", "rowspan", "align"},
    "th": {"colspan", "rowspan", "align", "scope"},
    "col": {"span", "width"},
    "colgroup": {"span"},
    "time": {"datetime"},
    "details": {"open"},
}

ALLOWED_URL_SCHEMES: set[str] = {"http", "https", "mailto", "data"}

# Stripped from CSS: remote fetches leak the viewer's IP/referrer and can be
# used as a side channel to exfiltrate what the artifact contains.
_CSS_IMPORT_RE = re.compile(r"@import[^;]+;?", re.IGNORECASE)
_CSS_REMOTE_URL_RE = re.compile(r"url\(\s*['\"]?\s*(?!data:)[a-z]*:?//[^)]*\)", re.IGNORECASE)
_CSS_EXPRESSION_RE = re.compile(r"expression\s*\(", re.IGNORECASE)

_DANGEROUS_TAG_RE = re.compile(
    r"<\s*(script|iframe|object|embed|form|input|button|base|link|meta|applet|frame|frameset)\b",
    re.IGNORECASE,
)
_EVENT_ATTR_RE = re.compile(r"\son[a-z]+\s*=", re.IGNORECASE)
_JS_URL_RE = re.compile(r"(javascript|vbscript)\s*:", re.IGNORECASE)


def _scrub_css(html: str) -> tuple[str, list[str]]:
    """Remove remote references from CSS while leaving the styling intact."""
    removed: list[str] = []
    if _CSS_IMPORT_RE.search(html):
        removed.append("css_import")
        html = _CSS_IMPORT_RE.sub("", html)
    if _CSS_REMOTE_URL_RE.search(html):
        removed.append("css_remote_url")
        html = _CSS_REMOTE_URL_RE.sub("none", html)
    if _CSS_EXPRESSION_RE.search(html):
        removed.append("css_expression")
        html = _CSS_EXPRESSION_RE.sub("none(", html)
    return html, removed


def sanitize_html(raw: str) -> tuple[str, dict]:
    """Return (safe_html, report).

    The report is persisted alongside the artifact so an operator can see what
    was stripped from a given generation without re-running the model.
    """
    findings: list[str] = []

    for label, pattern in (
        ("dangerous_tag", _DANGEROUS_TAG_RE),
        ("event_handler_attribute", _EVENT_ATTR_RE),
        ("javascript_url", _JS_URL_RE),
    ):
        if pattern.search(raw):
            findings.append(label)

    scrubbed, css_findings = _scrub_css(raw)
    findings.extend(css_findings)

    cleaned = nh3.clean(
        scrubbed,
        tags=ALLOWED_TAGS,
        attributes={k: set(v) for k, v in ALLOWED_ATTRIBUTES.items()},
        url_schemes=ALLOWED_URL_SCHEMES,
        # Without this, ammonia deletes the *contents* of <style> too, which
        # would silently gut every CSS artifact we generate.
        clean_content_tags={"script", "iframe", "object", "embed", "form"},
        link_rel="noopener noreferrer nofollow",
        strip_comments=True,
    )

    # data:text/html is a same-origin script vector that survives scheme checks.
    if "data:text/html" in cleaned.lower():
        findings.append("data_html_url")
        cleaned = re.sub(r"data:text/html[^\"'\s>]*", "", cleaned, flags=re.IGNORECASE)

    return cleaned, {
        "sanitizer": "nh3",
        "modified": cleaned != raw,
        "findings": sorted(set(findings)),
        "original_bytes": len(raw),
        "sanitized_bytes": len(cleaned),
    }


# Markdown is rendered client-side by a parser configured with HTML disabled, so
# raw tags inside markdown are displayed as text rather than parsed. We still
# strip the obvious script vector here as defence in depth for any consumer that
# renders the stored markdown with a laxer parser.
_MD_SCRIPT_RE = re.compile(r"<\s*script\b.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)


def sanitize_markdown(raw: str) -> tuple[str, dict]:
    findings: list[str] = []
    cleaned = raw
    if _MD_SCRIPT_RE.search(cleaned):
        findings.append("script_block")
        cleaned = _MD_SCRIPT_RE.sub("", cleaned)
    if _JS_URL_RE.search(cleaned):
        findings.append("javascript_url")
        cleaned = _JS_URL_RE.sub("", cleaned)
    return cleaned, {
        "sanitizer": "markdown-strip",
        "modified": cleaned != raw,
        "findings": sorted(set(findings)),
        "original_bytes": len(raw),
        "sanitized_bytes": len(cleaned),
    }


def sanitize_artifact(kind: str, content: str) -> tuple[str, dict]:
    return sanitize_html(content) if kind == "html" else sanitize_markdown(content)
