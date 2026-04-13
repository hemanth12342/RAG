"""
export.py - PDF export for chat conversations.
Uses fpdf2 (v2.x) with the XPos/YPos API.
All unicode is transliterated to latin-1 safe equivalents
since we use the core Helvetica font.
"""

from datetime import datetime
from typing import List, Dict, Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ── Colour palette ───────────────────────────────────────────────
_HEADER_BG  = (30,  30,  50)
_ACCENT     = (80,  80,  200)
_USER_BG    = (230, 240, 255)
_USER_FG    = (40,  80,  180)
_BOT_BG     = (240, 255, 240)
_BOT_FG     = (30,  130, 60)
_SRC_BG     = (248, 248, 248)
_SRC_FG     = (100, 100, 100)
_SNIPPET_FG = (140, 140, 140)
_FOOTER_FG  = (150, 150, 150)
_BODY_FG    = (30,  30,  30)

# Page margins
_L_MARGIN = 10
_R_MARGIN = 10


def _safe(text: str) -> str:
    """
    Transliterate common unicode punctuation to ASCII equivalents,
    then drop anything that cannot be encoded in latin-1.
    This is required when using fpdf2 with Helvetica (a core font).
    """
    table = str.maketrans({
        "\u2022": "-",     # bullet  •
        "\u2023": "-",     # triangular bullet
        "\u25cf": "-",     # black circle
        "\u25e6": "-",     # white bullet
        "\u2013": "-",     # en dash
        "\u2014": "--",    # em dash
        "\u2012": "-",     # figure dash
        "\u2015": "--",    # horizontal bar
        "\u2018": "'",     # left single quote
        "\u2019": "'",     # right single quote
        "\u201a": ",",     # single low quote
        "\u201c": '"',     # left double quote
        "\u201d": '"',     # right double quote
        "\u201e": '"',     # double low quote
        "\u2026": "...",   # ellipsis
        "\u00a0": " ",     # non-breaking space
        "\u00ad": "",      # soft hyphen
        "\u2192": "->",    # right arrow
        "\u27a2": ">",     # right arrow bullet
        "\u2714": "v",     # check mark
        "\u2716": "x",     # cross mark
        "\u00b7": "-",     # middle dot
    })
    text = text.translate(table)
    # Drop any remaining characters outside latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class _ChatPDF(FPDF):
    """fpdf2 subclass with branded header and page-number footer."""

    def header(self):
        self.set_fill_color(*_HEADER_BG)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        self.cell(
            0, 10,
            "RAG Document Chatbot - Conversation Export",
            align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_text_color(*_BODY_FG)
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_FOOTER_FG)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(
            0, 10,
            f"Generated on {now}  |  Page {self.page_no()}",
            align="C",
        )


def generate_chat_pdf(
    chat_history: List[Dict[str, Any]],
    document_names: List[str] = None,
) -> bytes:
    """
    Generate a styled PDF of the chat conversation.

    Args:
        chat_history:   List of {role, content, sources} dicts.
        document_names: File names of uploaded documents.

    Returns:
        PDF as bytes.
    """
    pdf = _ChatPDF()
    pdf.set_margins(_L_MARGIN, 10, _R_MARGIN)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Section: Uploaded Documents ──────────────────────────────
    _section_heading(pdf, "Uploaded Documents")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_BODY_FG)

    for name in (document_names or []):
        pdf.cell(
            0, 7,
            _safe(f"  - {name}"),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
    if not document_names:
        pdf.cell(
            0, 7, "  No documents recorded.",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
    pdf.ln(4)

    # ── Section: Conversation ─────────────────────────────────────
    _section_heading(pdf, "Conversation")

    q_num = 1
    for msg in chat_history:
        role    = msg.get("role", "unknown")
        content = _safe(msg.get("content", ""))
        sources = msg.get("sources", [])

        if role == "user":
            _render_bubble(pdf, f"  You  (Q{q_num})", content, _USER_BG, _USER_FG)
            q_num += 1
        else:
            _render_bubble(pdf, "  Assistant", content, _BOT_BG, _BOT_FG)
            if sources:
                _render_sources(pdf, sources)

        pdf.ln(4)

    return bytes(pdf.output())


# ── Private helpers ──────────────────────────────────────────────

def _usable_width(pdf: _ChatPDF) -> float:
    """Return printable page width in mm."""
    return pdf.w - pdf.l_margin - pdf.r_margin


def _section_heading(pdf: _ChatPDF, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_ACCENT)
    pdf.cell(
        0, 8, title,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_draw_color(*_ACCENT)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)


def _render_bubble(
    pdf: _ChatPDF,
    label: str,
    body: str,
    bg: tuple,
    fg: tuple,
):
    """Header bar + body text styled as a chat bubble."""
    width = _usable_width(pdf)

    # Header bar
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(*fg)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*fg)
    pdf.cell(
        width, 7,
        _safe(label),
        border="LTR", fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    # Body
    pdf.set_fill_color(*bg)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_BODY_FG)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        width, 6,
        body,
        fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    # Bottom border strip
    pdf.set_fill_color(*bg)
    pdf.cell(
        width, 3, "",
        border="LBR", fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )


def _render_sources(pdf: _ChatPDF, sources: List[Dict[str, str]]):
    """Source citation block below an assistant answer."""
    width = _usable_width(pdf)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*_SRC_FG)
    pdf.set_fill_color(*_SRC_BG)
    pdf.cell(
        width, 6, "  [Sources]",
        fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    for src in sources:
        name    = _safe(str(src.get("source", "N/A")))
        page    = _safe(str(src.get("page",   "N/A")))
        snippet = _safe(str(src.get("snippet", "")))

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*_SRC_FG)
        label = _safe(f"  - {name}  |  Page {page}")[:120]
        pdf.multi_cell(
            width, 5, label,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        if snippet:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*_SNIPPET_FG)
            snippet_text = _safe(f'  "{snippet[:160]}..."')
            pdf.multi_cell(
                width, 4, snippet_text,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
