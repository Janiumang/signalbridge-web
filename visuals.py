"""SignalBridge visual layer - brand CSS and result cards.

Matches the SignalBridge motion design: pale sage grid paper, navy ink,
deep-green accent, red reserved for alerts (death, the SIGNAL stamp),
bold grotesque headlines and spaced monospace eyebrows.

Everything here returns HTML strings rendered with st.html(). All values
coming from openFDA or the model are escaped before they reach the page.
"""

from __future__ import annotations

import base64
import datetime
import json
import math
from html import escape
from typing import Any

# ---------------------------------------------------------------------------
# Palette (sampled from the SignalBridge videos)
# ---------------------------------------------------------------------------

# Shared with VigilantAI (vigilant-ui/src/styles/tokens.css).
PAPER = "#f4f7f2"        # --bg-app
GRID = "#e7ece5"         # grid-paper lines (SignalBridge motion design)
CARD = "#ffffff"         # --bg-card
RULE = "#e2e8f0"         # --border / neutral-200
INK = "#0f172a"          # --text-primary
MUTED = "#475569"        # --text-secondary
FAINT = "#94a3b8"        # --text-muted
GREEN = "#2d5a4f"        # --brand / --color-primary
GREEN_DARK = "#1a3a31"   # --brand-dark
GREEN_MID = "#52796f"    # --accent / --color-secondary
GREEN_SOFT = "#e8f3f0"   # --color-primary-light
CHECK = "#059669"        # --color-success
RED = "#dc2626"          # --color-error
RED_SOFT = "#fee2e2"     # --color-error-light
AXIS = "#cbd5e1"         # neutral-300


def _img(svg: str, alt: str, cls: str = "", style: str = "") -> str:
    """Embed an SVG as a data-URI <img>.

    st.html() sanitizes inline <svg>, so graphics travel as images instead.
    """
    if 'xmlns=' not in svg[:200]:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    cls_attr = f' class="{cls}"' if cls else ""
    style_attr = f' style="{style}"' if style else ""
    return f'<img{cls_attr}{style_attr} alt="{escape(alt)}" src="data:image/svg+xml;base64,{data}">'


GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sora:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{
  --sb-paper:{PAPER}; --sb-ink:{INK}; --sb-muted:{MUTED}; --sb-faint:{FAINT};
  --sb-green:{GREEN}; --sb-green-soft:{GREEN_SOFT}; --sb-red:{RED};
  --sb-rule:{RULE}; --sb-card:{CARD};
  --sb-sans:'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --sb-display:'Sora', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --sb-mono:'JetBrains Mono', 'Fira Code', monospace;
}}

/* Grid-paper background */
.stApp {{
  background-color: var(--sb-paper);
  background-image:
    linear-gradient({GRID} 1px, transparent 1px),
    linear-gradient(90deg, {GRID} 1px, transparent 1px);
  background-size: 48px 48px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: transparent; }}

html, body, .stApp, .stMarkdown, p, li, label, button, input, textarea {{
  font-family: var(--sb-sans);
  color: var(--sb-ink);
}}
.stMarkdown p, .stMarkdown li {{ font-size: 1.02rem; line-height: 1.6; }}
h1, h2, h3, h4 {{ font-family: var(--sb-display) !important; color: var(--sb-ink); letter-spacing: -0.02em; }}
code, pre, .stCode {{ font-family: var(--sb-mono) !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 880px; }}

/* Alerts (free-tier status) in brand tones */
[data-testid="stAlertContainer"], [data-testid="stAlert"] > div {{ background: {GREEN_SOFT} !important; color: {GREEN_DARK} !important; border: 1px solid #cfe3dc; border-radius: 8px; }}
[data-testid="stAlertContainer"] p, [data-testid="stAlert"] p {{ color: {GREEN_DARK} !important; }}
[data-testid="stAlertContentWarning"] p {{ color: #92400e !important; }}
/* Chat avatars in brand colours */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"] {{ background: {GREEN} !important; color: #fff !important; }}
[data-testid="stChatMessageAvatarUser"] {{ background: {GREEN_MID} !important; }}
[data-testid="stChatInput"] > div {{ background: var(--sb-card) !important; }}

/* Sign-up form */
[data-testid="stForm"] {{ border: none; padding: 0; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: rgba(255,255,255,.55); border-color: var(--sb-rule) !important; border-radius: 12px; }}
[data-testid="stFormSubmitButton"] button {{ background: var(--sb-green) !important; color: #fff !important; border: none !important; font-family: var(--sb-sans) !important; font-weight: 600; font-size: .95rem; }}
[data-testid="stFormSubmitButton"] button p {{ color: #fff !important; font-family: var(--sb-sans) !important; }}
[data-testid="stFormSubmitButton"] button:hover {{ background: {GREEN_DARK} !important; color: #fff !important; }}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {{ background: var(--sb-card) !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: #eef1ea; border-right: 1px solid var(--sb-rule); }}

/* Example prompts + buttons: mono chips */
.stButton > button {{
  background: var(--sb-card); color: var(--sb-ink);
  border: 1px solid var(--sb-rule); border-radius: 8px;
  font-family: var(--sb-mono); font-size: 0.8rem; text-align: left;
  padding: 0.7rem 0.9rem; line-height: 1.45; box-shadow: none;
}}
.stButton > button:hover {{ border-color: var(--sb-green); color: var(--sb-green); background: var(--sb-card); }}
.stButton > button:focus:not(:active) {{ border-color: var(--sb-green); color: var(--sb-green); }}

/* Chat */
[data-testid="stChatMessage"] {{ background: transparent; padding: 0.4rem 0; }}
[data-testid="stChatInput"] {{ border-radius: 10px; border: 1px solid var(--sb-rule); background: var(--sb-card); }}
[data-testid="stChatInput"] textarea {{ font-size: 1rem; }}
[data-testid="stExpander"] details {{ background: var(--sb-card); border: 1px solid var(--sb-rule); border-radius: 8px; }}
[data-testid="stExpander"] summary p {{ font-family: var(--sb-mono); font-size: 0.78rem; letter-spacing: 0.06em; color: var(--sb-muted); }}

/* Brand bits */
.sb-bar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 2.2rem; }}
.sb-brand {{ font-family: var(--sb-display); font-weight: 800; font-size: 1.15rem; letter-spacing:-0.01em; display:flex; align-items:center; gap:.55rem; color:var(--sb-ink); }}
.sb-dot {{ width:11px; height:11px; border-radius:50%; background:var(--sb-green); display:inline-block; }}
.sb-side {{ margin-bottom: 1rem; }}
.sb-side .sb-src {{ display:none; }}
.sb-src {{ font-family: var(--sb-mono); font-size: .78rem; color: var(--sb-muted); }}
.sb-eyebrow {{ font-family: var(--sb-mono); font-size: .74rem; letter-spacing: .22em; text-transform: uppercase; color: var(--sb-green); font-weight:500; margin: 0 0 .7rem; }}
.sb-hero {{ font-family: var(--sb-display); font-weight: 800; font-size: clamp(2.4rem, 7vw, 4.1rem); line-height: .98; letter-spacing: -0.035em; color: var(--sb-ink); margin: 0 0 .9rem; }}
.sb-hero em {{ font-style: normal; color: var(--sb-green); }}
.sb-sub {{ font-size: 1.15rem; color: var(--sb-muted); margin: 0 0 1.6rem; }}
.sb-pulse {{ display:block; margin: -.4rem 0 1.4rem auto; }}
.sb-chips {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.2rem 0 1.4rem; }}
.sb-chip {{ font-family: var(--sb-mono); font-size: .74rem; background: var(--sb-card); border:1px solid var(--sb-rule); border-radius: 6px; padding: .3rem .6rem; color: var(--sb-ink); }}
.sb-chip-cmp {{ background: {GREEN_SOFT}; border-color: #cfe3dc; color: {GREEN_DARK}; }}
.sb-more {{ margin: .4rem 0 .9rem; }}
.sb-more summary {{ font-family: var(--sb-mono); font-size: .76rem; color: var(--sb-muted); cursor: pointer; padding: .5rem 0; letter-spacing: .04em; }}
.sb-arrow {{ font-family: var(--sb-mono); color: var(--sb-faint); font-size:.8rem; }}

/* Cards */
.sb-card {{ background: rgba(255,255,255,.55); border: 1px solid var(--sb-rule); border-radius: 12px; padding: 1.3rem 1.4rem 1.2rem; margin: .9rem 0; position:relative; overflow:hidden; }}
.sb-h {{ font-family: var(--sb-display); font-weight: 800; font-size: 1.45rem; letter-spacing:-0.02em; margin: 0 0 1rem; color: var(--sb-ink); line-height:1.15; }}
.sb-note {{ font-family: var(--sb-mono); font-size: .72rem; color: var(--sb-muted); margin-top: .8rem; line-height:1.5; }}

/* Plan rows */
.sb-row {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; background: var(--sb-card); border-left: 3px solid var(--sb-green); padding: .55rem .8rem; margin-bottom: .4rem; font-family: var(--sb-mono); font-size: .8rem; }}
.sb-row > span:first-child {{ flex:1; min-width:0; overflow-wrap:anywhere; }}
.sb-row .t {{ color: var(--sb-ink); font-weight: 500; }}
.sb-row .a {{ color: var(--sb-faint); }}
.sb-row .r {{ color: var(--sb-ink); white-space: nowrap; }}
.sb-row.err {{ border-left-color: var(--sb-red); }}

/* ROR */
.sb-formula {{ font-family: var(--sb-mono); font-size: 1.15rem; color: var(--sb-muted); margin-bottom:.2rem; }}
.sb-formula b {{ color: var(--sb-ink); font-weight:500; }}
.sb-big {{ display:flex; align-items:flex-end; gap: 1.1rem; flex-wrap: wrap; }}
.sb-num {{ font-family: var(--sb-display); font-weight: 800; font-size: clamp(3.6rem, 12vw, 6rem); line-height: .9; letter-spacing: -0.04em; color: var(--sb-green); }}
.sb-ci {{ font-family: var(--sb-mono); font-size: .95rem; color: var(--sb-muted); padding-bottom: .5rem; line-height:1.45; }}
.sb-two {{ display:grid; grid-template-columns: 1fr 1fr; gap: 1.6rem; margin-top: 1rem; }}
@media (max-width: 640px) {{ .sb-two {{ grid-template-columns: 1fr; }} }}
.sb-tablewrap {{ overflow-x:auto; }}
.sb-table {{ width:100%; border-collapse: collapse; font-family: var(--sb-mono); }}
.sb-table th {{ font-size: .66rem; letter-spacing:.14em; text-transform: uppercase; color: var(--sb-faint); font-weight:500; text-align:right; padding: .2rem .5rem; }}
.sb-table td {{ text-align:right; padding: .45rem .5rem; font-size: 1rem; color: var(--sb-ink); border-bottom: 1px solid var(--sb-rule); }}
.sb-table td.lab {{ text-align:left; font-size: .76rem; color: var(--sb-muted); }}
.sb-table td.hit {{ background: var(--sb-green-soft); color: var(--sb-green); font-weight:700; }}
.sb-table sup {{ color: var(--sb-faint); font-size: .6rem; margin-right: .35rem; }}
.sb-crit {{ display:flex; align-items:center; gap: .7rem; padding: .5rem 0; border-bottom: 1px solid var(--sb-rule); }}
.sb-crit .lbl {{ flex:1; font-size: .98rem; }}
.sb-crit .val {{ font-family: var(--sb-mono); font-weight: 700; font-size: .95rem; }}
.sb-stamp {{ position:absolute; right: 1.6rem; top: 4.6rem; transform: rotate(-8deg); border: 4px solid var(--sb-red); color: var(--sb-red); border-radius: 8px; padding: .15rem .8rem; font-weight: 800; font-size: 2rem; letter-spacing: .04em; font-family: var(--sb-display); opacity:.92; }}
.sb-stamp.no {{ border-color: var(--sb-faint); color: var(--sb-faint); font-size: 1.3rem; }}
@media (max-width: 640px) {{ .sb-stamp {{ position: static; display:inline-block; margin: .6rem 0; }} .sb-table td {{ font-size:.8rem; padding:.4rem .3rem; }} .sb-table th {{ padding:.2rem .3rem; }} .sb-card {{ padding: 1.1rem 1rem; }} }}

/* Bars */
.sb-bars {{ display:grid; grid-template-columns: minmax(110px, 34%) 1fr auto; gap: .5rem .8rem; align-items:center; }}
.sb-bars .l {{ font-size: .9rem; color: var(--sb-ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.sb-bars .track {{ background: var(--sb-card); height: 16px; border-radius: 2px; }}
.sb-bars .fill {{ background: var(--sb-green); height: 16px; border-radius: 2px; }}
.sb-bars .fill.red {{ background: var(--sb-red); }}
.sb-bars .n {{ font-family: var(--sb-mono); font-size: .88rem; text-align:right; min-width: 3.5rem; }}
.sb-alert {{ background: {RED_SOFT}; color: var(--sb-red); border-radius: 6px; padding: .55rem .8rem; font-weight:600; font-size:.88rem; margin-top: .9rem; }}
</style>
"""


# ---------------------------------------------------------------------------
# Static page pieces
# ---------------------------------------------------------------------------

PULSE_SVG = (
    f'<svg class="sb-pulse" width="260" height="90" viewBox="0 0 260 90" aria-hidden="true">'
    f'<polyline fill="none" stroke="{GREEN}" stroke-width="2.5" '
    f'points="0,62 75,62 88,18 102,82 116,6 132,62 260,62"/></svg>'
)


def brand_bar() -> str:
    return (
        '<div class="sb-bar"><div class="sb-brand"><span class="sb-dot"></span>'
        'SignalBridge openFDA</div><div class="sb-src">openFDA · FAERS</div></div>'
    )


def hero() -> str:
    flow = ["Plain language", "MedDRA PT", "2×2 table", "ROR · CI · flag"]
    chips = '<span class="sb-arrow">→</span>'.join(
        f'<span class="sb-chip">{escape(s)}</span>' for s in flow
    )
    return (
        '<div class="sb-eyebrow">Pharmacovigilance search</div>'
        '<div class="sb-hero">Ask FAERS<br>a <em>safety question.</em></div>'
        '<div class="sb-sub">Get the signal math back in seconds.</div>'
        f'<div class="sb-chips">{chips}</div>'
    )


# ---------------------------------------------------------------------------
# Tool-call "plan" rows
# ---------------------------------------------------------------------------

_SKIP_ARGS = {"drug_name"}


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ratio(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _args_summary(inp: dict[str, Any]) -> str:
    parts = []
    for k, v in inp.items():
        if v in (None, "", [], {}) or k in _SKIP_ARGS:
            continue
        if k == "min_age":
            parts.append(f"age ≥ {v}")
        elif k == "max_age":
            parts.append(f"age ≤ {v}")
        elif k == "start_date":
            parts.append(f"since {v}")
        elif k == "end_date":
            parts.append(f"to {v}")
        else:
            parts.append(f"{k} = {v}")
    return "  ".join(parts)


def _result_summary(name: str, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if "error" in data:
        return "error"
    if name == "calculate_reporting_odds_ratio":
        return f"ROR {_fmt_ratio(data.get('ror'))}"
    if name in ("count_adverse_events", "count_reactions"):
        counts = data.get("counts") or []
        pivot = data.get("pivot", "")
        if pivot == "year" and counts:
            labels = [str(c.get("label")) for c in counts]
            return f"{labels[0]}–{labels[-1]}"
        return f"{len(counts)} {('flags' if pivot == 'seriousness_subtype' else 'groups')}"
    for key in ("total_matching", "total", "total_results"):
        if key in data:
            noun = "recalls" if "recall" in name else "reports"
            return f"{_fmt_int(data[key])} {noun}"
    if name == "get_drug_label":
        return "label"
    return ""


_FILTER_KEYS = ("drug_name", "reaction", "min_age", "max_age", "country", "start_date", "end_date")


def _age_label(lo: Any, hi: Any) -> str | None:
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None:
        return f"age {lo}–{hi}"
    if lo is not None:
        return f"age ≥ {lo}"
    return f"age ≤ {hi}"


def _frame(tool_calls: list[dict[str, Any]]) -> tuple[list[str], list[str], bool]:
    """Shared filter chips, subgroup 'compare' chips, and whether every
    call used the same frame."""
    inputs = [c.get("input") or {} for c in tool_calls]
    values: dict[str, set] = {k: set() for k in _FILTER_KEYS}
    for inp in inputs:
        for k in _FILTER_KEYS:
            if inp.get(k) not in (None, ""):
                values[k].add(str(inp[k]))

    chips: list[str] = []
    if len(values["drug_name"]) == 1:
        chips.append(f"drug {next(iter(values['drug_name']))}")
    if len(values["reaction"]) == 1:
        chips.append(f"PT {next(iter(values['reaction']))}")
    if len(values["min_age"]) <= 1 and len(values["max_age"]) <= 1:
        lo = next(iter(values["min_age"]), None)
        hi = next(iter(values["max_age"]), None)
        age = _age_label(lo, hi)
        if age:
            chips.append(age.replace("age", "age", 1))
    if len(values["country"]) == 1:
        chips.append(f"country {next(iter(values['country']))}")
    if len(values["start_date"]) == 1:
        chips.append(f"received ≥ {next(iter(values['start_date']))}")
    if len(values["end_date"]) == 1:
        chips.append(f"received ≤ {next(iter(values['end_date']))}")

    compare: list[str] = []
    if len(values["drug_name"]) > 1:
        compare.append("compare " + " vs ".join(sorted(values["drug_name"])))
    if len(values["min_age"]) > 1 or len(values["max_age"]) > 1:
        groups = sorted(
            {(_age_label(i.get("min_age"), i.get("max_age")) or "all ages") for i in inputs
             if i.get("min_age") is not None or i.get("max_age") is not None}
        )
        compare.append("compare " + " vs ".join(groups))
    if len(values["country"]) > 1:
        compare.append("compare " + " vs ".join(sorted(values["country"])))

    def sig(i: dict[str, Any]) -> tuple:
        return tuple(i.get(k) for k in ("drug_name", "country", "start_date", "end_date"))

    consistent = len(tool_calls) > 1 and len({sig(i) for i in inputs}) == 1
    return chips, compare, consistent


def frame_chips(tool_calls: list[dict[str, Any]]) -> str:
    chips, compare, _ = _frame(tool_calls)
    if not chips and not compare:
        return ""
    html = "".join(f'<span class="sb-chip">{escape(c)}</span>' for c in chips)
    html += "".join(f'<span class="sb-chip sb-chip-cmp">{escape(c)}</span>' for c in compare)
    return f'<div class="sb-chips" style="margin:.1rem 0 1rem">{html}</div>'


def plan_rows(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return ""
    n = len(tool_calls)
    # Filters shared by every call are shown once as chips; rows show only
    # what distinguishes each call (pivot, subgroup).
    inputs = [c.get("input") or {} for c in tool_calls]
    shared = {
        k for k in _FILTER_KEYS
        if n > 1 and all(k in i for i in inputs) and len({str(i.get(k)) for i in inputs}) == 1
    }
    rows = []
    for call in tool_calls:
        data = call.get("data")
        summary = _result_summary(call["name"], data)
        err = " err" if summary == "error" else ""
        own = {k: v for k, v in (call.get("input") or {}).items() if k not in shared}
        own_txt = _args_summary(own).replace("pivot = ", "pivot = ")
        rows.append(
            f'<div class="sb-row{err}"><span><span class="t">{escape(call["name"])}</span> '
            f'<span class="a">{escape(own_txt)}</span></span>'
            f'<span class="r">{escape(summary)}</span></div>'
        )
    chips, _, consistent = _frame(tool_calls)
    if n == 1:
        title = "One live query."
    elif consistent:
        title = f"{n} tool calls, one consistent frame."
    else:
        title = f"{n} live queries."
    footer = ""
    if consistent and chips:
        footer = (
            '<div class="sb-note">Same filters on every call · '
            + escape(" · ".join(c for c in chips if not c.startswith("drug ")))
            + "</div>"
        )
    return (
        '<div class="sb-card"><div class="sb-eyebrow">The plan it ran</div>'
        f'<div class="sb-h">{title}</div>{frame_chips(tool_calls)}{"".join(rows)}{footer}</div>'
    )


# ---------------------------------------------------------------------------
# ROR card
# ---------------------------------------------------------------------------

_LOG_MIN, _LOG_MAX = math.log(0.5), math.log(10)


def _x(v: float, x0: float, x1: float) -> float:
    v = min(max(v, 0.5), 10)
    return x0 + (math.log(v) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN) * (x1 - x0)


def forest_strip(ror: float | None, lo: float | None, hi: float | None) -> str:
    """Log-scale 0.5–10 axis with 'no association' line at 1 and the CI bar."""
    w, x0, x1, y = 640, 24, 616, 70
    ticks = "".join(
        f'<line x1="{_x(t, x0, x1):.1f}" y1="{y - 9}" x2="{_x(t, x0, x1):.1f}" y2="{y + 9}" stroke="{FAINT}" stroke-width="1.5"/>'
        f'<text x="{_x(t, x0, x1):.1f}" y="{y + 30}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="12" fill="{MUTED}">{t:g}</text>'
        for t in (0.5, 1, 2, 5, 10)
    )
    one = _x(1, x0, x1)
    mark = ""
    if ror and ror > 0:
        lo_v = lo if lo else ror
        hi_v = hi if hi else ror
        xl, xh, xp = _x(lo_v, x0, x1), _x(hi_v, x0, x1), _x(ror, x0, x1)
        mark = (
            f'<line x1="{xl:.1f}" y1="{y}" x2="{max(xh, xl + 4):.1f}" y2="{y}" stroke="{GREEN}" stroke-width="7" stroke-linecap="round"/>'
            f'<rect x="{xp - 6:.1f}" y="{y - 11}" width="12" height="22" rx="2" fill="{GREEN}"/>'
        )
        if ror > 10 or (hi and hi > 10):
            mark += f'<text x="{x1}" y="{y - 16}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="11" fill="{MUTED}">→ beyond 10</text>'
    svg = (
        f'<svg viewBox="0 0 {w} 112" width="{w}" height="112">'
        f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{AXIS}" stroke-width="3"/>'
        f'<line x1="{one:.1f}" y1="8" x2="{one:.1f}" y2="{y + 9}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 6"/>'
        f'<text x="{one + 8:.1f}" y="20" font-family="JetBrains Mono, monospace" font-size="12" fill="{MUTED}">no association</text>'
        f"{ticks}{mark}</svg>"
    )
    return _img(svg, "Reporting odds ratio on a log scale, with the 95% confidence interval", style="width:100%;height:auto;display:block;margin:.4rem 0 .2rem")


def _check(ok: bool | None) -> str:
    return _img(_check_svg(ok), "met" if ok else "not met", style="width:30px;height:30px;flex:none")


def _check_svg(ok: bool | None) -> str:
    if ok:
        return (
            f'<svg width="30" height="30" viewBox="0 0 30 30"><circle cx="15" cy="15" r="13" fill="none" stroke="{CHECK}" stroke-width="2"/>'
            f'<polyline points="10,15.5 13.5,19 20,11.5" fill="none" stroke="{CHECK}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
    return (
        f'<svg width="30" height="30" viewBox="0 0 30 30"><circle cx="15" cy="15" r="13" fill="none" stroke="{FAINT}" stroke-width="2"/>'
        f'<line x1="10.5" y1="10.5" x2="19.5" y2="19.5" stroke="{FAINT}" stroke-width="2.2" stroke-linecap="round"/>'
        f'<line x1="19.5" y1="10.5" x2="10.5" y2="19.5" stroke="{FAINT}" stroke-width="2.2" stroke-linecap="round"/></svg>'
    )


def ror_card(data: dict[str, Any]) -> str:
    if not isinstance(data, dict) or "contingency_table" not in data:
        return ""
    t = data["contingency_table"]
    a, b = t.get("a_drug_and_reaction"), t.get("b_drug_no_reaction")
    c, d = t.get("c_reaction_no_drug"), t.get("d_neither")
    ror, lo, hi = data.get("ror"), data.get("ci_95_lower"), data.get("ci_95_upper")
    signal = data.get("signal")
    drug = escape(str(data.get("drug_name", "")))
    reaction = escape(str(data.get("reaction", "")))

    stamp = ""
    if signal is True:
        stamp = '<div class="sb-stamp">SIGNAL</div>'
    elif signal is False:
        stamp = '<div class="sb-stamp no">NO SIGNAL</div>'

    ci_txt = (
        f"95% CI<br>{_fmt_ratio(lo)} – {_fmt_ratio(hi)}" if lo is not None and hi is not None
        else "95% CI<br>undefined"
    )

    crit = [
        ("ROR above 1.0", ror is not None and ror > 1, _fmt_ratio(ror)),
        ("CI lower bound above 1.0", lo is not None and lo > 1, _fmt_ratio(lo)),
        ("At least 3 co-reports", (a or 0) >= 3, _fmt_int(a)),
    ]
    crit_html = "".join(
        f'<div class="sb-crit">{_check(ok)}<span class="lbl">{escape(lbl)}</span><span class="val">{escape(val)}</span></div>'
        for lbl, ok, val in crit
    )

    filters = data.get("filters_applied") or {}
    scope = _args_summary({k: v for k, v in filters.items() if v is not None}) or "all FAERS, no filters"

    exp = data.get("reaction_expansion")
    exp_note = ""
    if isinstance(exp, dict) and exp.get("expanded_to_meddra_pts"):
        exp_note = (
            "<br>Reaction grouped from a lay term: "
            + escape(", ".join(exp["expanded_to_meddra_pts"]))
            + "."
        )

    table = (
        '<div class="sb-tablewrap"><table class="sb-table"><tr><th></th>'
        "<th>Reaction</th><th>No reaction</th></tr>"
        f'<tr><td class="lab">{drug or "Drug"}</td><td class="hit"><sup>a</sup>{_fmt_int(a)}</td><td><sup>b</sup>{_fmt_int(b)}</td></tr>'
        f'<tr><td class="lab">All other</td><td><sup>c</sup>{_fmt_int(c)}</td><td><sup>d</sup>{_fmt_int(d)}</td></tr></table></div>'
    )

    return (
        f'<div class="sb-card">{stamp}'
        f'<div class="sb-eyebrow">Reporting odds ratio · {drug} × {reaction}</div>'
        '<div class="sb-formula">ROR = (<b>a</b> × <b>d</b>) ÷ (<b>b</b> × <b>c</b>)</div>'
        f'<div class="sb-big"><div class="sb-num">{_fmt_ratio(ror)}</div><div class="sb-ci">{ci_txt}</div></div>'
        f"{forest_strip(ror, lo, hi)}"
        f'<div class="sb-two"><div><div class="sb-eyebrow">Four counts, one sampling frame</div>{table}</div>'
        f'<div><div class="sb-eyebrow">Signal criteria</div>{crit_html}</div></div>'
        f'<div class="sb-note">Scope: {escape(scope)}.{exp_note}<br>'
        "Disproportionality flags reporting patterns, not incidence or causality. Clinical review still decides.</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Count cards
# ---------------------------------------------------------------------------

_PIVOT_TITLES = {
    "reaction": "Top reported reactions",
    "country": "Reports by country",
    "reporter_qualification": "Who files the reports",
    "concomitant_drug": "Most common co-reported drugs",
    "seriousness_subtype": "Seriousness of the reports",
    "year": "Reports per year",
}


def _year_columns(counts: list[dict[str, Any]]) -> str:
    vals = [(str(c.get("label")), int(c.get("count") or 0)) for c in counts]
    if not vals:
        return ""
    top = max(v for _, v in vals) or 1
    n = len(vals)
    w, h, pad, base = 640, 220, 6, 190
    bw = (w - pad * (n + 1)) / n
    this_year = str(datetime.date.today().year)
    defs = (
        '<defs><pattern id="sbhatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="8" height="8" fill="{GREEN_SOFT}"/><line x1="0" y1="0" x2="0" y2="8" stroke="{GREEN}" stroke-width="3"/></pattern></defs>'
    )
    bars = []
    for i, (lab, v) in enumerate(vals):
        x = pad + i * (bw + pad)
        bh = max(1.0, v / top * 150)
        fill = "url(#sbhatch)" if lab == this_year else GREEN_MID
        short = lab[-2:] if n > 8 else lab
        if lab == this_year:
            short += "*"
        bars.append(
            f'<rect x="{x:.1f}" y="{base - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{fill}"/>'
            f'<text x="{x + bw / 2:.1f}" y="{base - bh - 6:.1f}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="{11 if n <= 12 else 9}" fill="{INK}">{v:,}</text>'
            f'<text x="{x + bw / 2:.1f}" y="{base + 18}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="{MUTED}">{escape(short)}</text>'
        )
    note = (
        f'<div class="sb-note">*{this_year} is a partial year.</div>'
        if any(lab == this_year for lab, _ in vals) else ""
    )
    svg = f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{defs}{"".join(bars)}</svg>'
    return _img(svg, "Reports per year", style="width:100%;height:auto;display:block") + note


def _hbars(pivot: str, counts: list[dict[str, Any]], limit: int = 10) -> str:
    rows = [(str(c.get("label") or "Unknown"), int(c.get("count") or 0)) for c in counts[:limit]]
    if not rows:
        return '<div class="sb-note">No matching reports.</div>'
    top = max(v for _, v in rows) or 1
    out = []
    for lab, v in rows:
        red = " red" if (pivot == "seriousness_subtype" and lab == "Death") or (
            pivot == "reporter_qualification" and lab.lower() == "lawyer"
        ) else ""
        pct = max(1.5, v / top * 100) if v else 0
        out.append(
            f'<div class="l" title="{escape(lab)}">{escape(lab)}</div>'
            f'<div class="track"><div class="fill{red}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="n">{v:,}</div>'
        )
    return f'<div class="sb-bars">{"".join(out)}</div>'


def count_card(data: dict[str, Any]) -> str:
    if not isinstance(data, dict) or "counts" not in data or "error" in data:
        return ""
    pivot = data.get("pivot", "reaction")
    counts = data.get("counts") or []
    total = data.get("total_matching")
    drug = escape(str(data.get("drug_name", "")))
    eyebrow = f"{escape(pivot.replace('_', ' '))} · {_fmt_int(total)} reports · {drug}"
    body = _year_columns(counts) if pivot == "year" else _hbars(pivot, counts)

    extra = ""
    if pivot == "seriousness_subtype":
        extra = '<div class="sb-note">A report can carry several flags. A death flag records the outcome, not causality.</div>'
    if pivot == "reporter_qualification" and total:
        lawyer = next((c for c in counts if str(c.get("label", "")).lower() == "lawyer"), None)
        if lawyer and lawyer.get("count") and lawyer["count"] / total >= 0.05:
            extra = (
                f'<div class="sb-alert">{lawyer["count"]:,} of {total:,} filed by lawyers: '
                "check for stimulated reporting</div>"
            )
    exp = data.get("reaction_expansion")
    if isinstance(exp, dict) and exp.get("expanded_to_meddra_pts"):
        extra += (
            '<div class="sb-note">Reaction grouped from a lay term: '
            + escape(", ".join(exp["expanded_to_meddra_pts"])) + ".</div>"
        )

    return (
        f'<div class="sb-card"><div class="sb-eyebrow">{eyebrow}</div>'
        f'<div class="sb-h">{escape(_PIVOT_TITLES.get(pivot, "Counts"))}</div>{body}{extra}</div>'
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _group_label(data: dict[str, Any], vary: set[str]) -> str:
    f = data.get("filters_applied") or {}
    parts = []
    if "drug" in vary:
        parts.append(str(data.get("drug_name", "")))
    if "reaction" in vary:
        parts.append(str(data.get("reaction", "")))
    if "age" in vary:
        lo, hi = f.get("min_age"), f.get("max_age")
        if lo is not None and hi is None:
            parts.append(f"Adults {lo}+")
        elif lo is not None and hi is not None:
            parts.append(f"Adults {lo}–{hi}")
        elif hi is not None:
            parts.append(f"Age ≤ {hi}")
        else:
            parts.append("All ages")
    if "country" in vary:
        parts.append(str(f.get("country") or "All countries"))
    if "dates" in vary:
        parts.append(f"{f.get('start_date') or '…'} to {f.get('end_date') or 'now'}")
    return " · ".join(p for p in parts if p) or "Group"


def ror_compare_card(datas: list[dict[str, Any]]) -> str:
    """Several RORs on one log axis - e.g. 'The signal holds in both groups.'"""
    datas = [d for d in datas if isinstance(d, dict) and "contingency_table" in d]
    if len(datas) < 2:
        return ""

    def fa(d: dict[str, Any], k: str) -> Any:
        return (d.get("filters_applied") or {}).get(k)

    vary: set[str] = set()
    if len({str(d.get("drug_name", "")).lower() for d in datas}) > 1:
        vary.add("drug")
    if len({str(d.get("reaction", "")).lower() for d in datas}) > 1:
        vary.add("reaction")
    if len({(fa(d, "min_age"), fa(d, "max_age")) for d in datas}) > 1:
        vary.add("age")
    if len({fa(d, "country") for d in datas}) > 1:
        vary.add("country")
    if len({(fa(d, "start_date"), fa(d, "end_date")) for d in datas}) > 1:
        vary.add("dates")

    signals = [d.get("signal") for d in datas]
    if all(s is True for s in signals):
        title = "The signal holds in both groups." if len(datas) == 2 else "The signal holds in every group."
    elif any(s is True for s in signals):
        title = "The signal differs between groups."
    else:
        title = "No signal in any group."

    w, label_w, x0, x1 = 640, 0, 190, 600
    row_h, top = 58, 34
    h = top + row_h * len(datas) + 46
    mono = "JetBrains Mono, monospace"
    sans = "Plus Jakarta Sans, sans-serif"
    one = _x(1, x0, x1)
    parts = [
        f'<line x1="{one:.1f}" y1="10" x2="{one:.1f}" y2="{h - 40}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 5"/>',
        f'<text x="{one + 6:.1f}" y="20" font-family="{mono}" font-size="11" fill="{MUTED}">no association</text>',
    ]
    for i, d in enumerate(datas):
        y = top + i * row_h + 22
        label = _group_label(d, vary)
        a = (d.get("contingency_table") or {}).get("a_drug_and_reaction")
        parts.append(f'<text x="0" y="{y - 2}" font-family="{mono}" font-size="14" fill="{INK}">{escape(label[:24])}</text>')
        parts.append(f'<text x="0" y="{y + 16}" font-family="{mono}" font-size="11" fill="{FAINT}">n = {_fmt_int(a)}</text>')
        ror, lo, hi = d.get("ror"), d.get("ci_95_lower"), d.get("ci_95_upper")
        if ror and ror > 0:
            xl, xh, xp = _x(lo or ror, x0, x1), _x(hi or ror, x0, x1), _x(ror, x0, x1)
            col = GREEN if d.get("signal") else FAINT
            parts.append(f'<line x1="{xl:.1f}" y1="{y}" x2="{max(xh, xl + 4):.1f}" y2="{y}" stroke="{col}" stroke-width="7" stroke-linecap="round"/>')
            parts.append(f'<rect x="{xp - 6:.1f}" y="{y - 11}" width="12" height="22" rx="2" fill="{col}"/>')
            txt = f"{_fmt_ratio(ror)} ({_fmt_ratio(lo)}–{_fmt_ratio(hi)})"
            tx = min(xp + 14, x1 - 4)
            anchor = "start" if xp + 14 < x1 - 120 else "end"
            if anchor == "end":
                tx = xl - 10
            parts.append(f'<text x="{tx:.1f}" y="{y + 5}" text-anchor="{anchor}" font-family="{mono}" font-size="13" fill="{INK}">{escape(txt)}</text>')
        else:
            parts.append(f'<text x="{x0}" y="{y + 5}" font-family="{mono}" font-size="12" fill="{MUTED}">ROR undefined</text>')
    ay = h - 34
    parts.append(f'<line x1="{x0}" y1="{ay}" x2="{x1}" y2="{ay}" stroke="{AXIS}" stroke-width="2"/>')
    for t in (0.5, 1, 2, 5, 10):
        tx = _x(t, x0, x1)
        parts.append(f'<line x1="{tx:.1f}" y1="{ay - 6}" x2="{tx:.1f}" y2="{ay + 6}" stroke="{FAINT}" stroke-width="1.5"/>')
        parts.append(f'<text x="{tx:.1f}" y="{ay + 24}" text-anchor="middle" font-family="{mono}" font-size="11" fill="{MUTED}">{t:g}</text>')
    svg = f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{"".join(parts)}</svg>'

    eyebrow = "Stratified ROR" if "age" in vary else "ROR comparison"
    return (
        f'<div class="sb-card"><div class="sb-eyebrow">{eyebrow}</div><div class="sb-h">{escape(title)}</div>'
        + _img(svg, "Reporting odds ratios by group on a log scale", style="width:100%;height:auto;display:block")
        + '<div class="sb-note">Compare groups only within the same frame: background reporting rates differ by '
        "subgroup, so a smaller ROR can mean the reaction is common across all drugs in that group, not that the "
        "drug is safer there.</div></div>"
    )


def result_cards(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Visual cards for the tool results worth drawing (ROR + counts)."""
    cards: list[str] = []
    rors = [c.get("data") for c in tool_calls if c["name"] == "calculate_reporting_odds_ratio"]
    rors = [d for d in rors if isinstance(d, dict) and "contingency_table" in d]
    if len(rors) >= 2:
        cards.append(ror_compare_card(rors))
        # Detail (2x2 + criteria) for the first group; the rest stay folded.
        cards.append(ror_card(rors[0]))
        more = "".join(ror_card(d) for d in rors[1:])
        cards.append(
            f'<details class="sb-more"><summary>Full 2×2 and criteria for the other '
            f'{len(rors) - 1} group{"s" if len(rors) > 2 else ""}</summary>{more}</details>'
        )
    elif rors:
        cards.append(ror_card(rors[0]))

    order = {"seriousness_subtype": 0, "reporter_qualification": 1, "year": 2}
    counts = [c for c in tool_calls if c["name"] in ("count_adverse_events", "count_reactions")]
    counts.sort(key=lambda c: order.get((c.get("data") or {}).get("pivot", ""), 9))
    for call in counts:
        html = count_card(call.get("data"))
        if html:
            cards.append(html)
    return [c for c in cards if c]
