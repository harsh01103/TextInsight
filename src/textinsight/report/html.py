"""A self-contained HTML report: no scripts, and nothing loaded from elsewhere.

Every value from the analysed text is escaped, because that text is untrusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

__all__ = ["render_html"]

_MAX_ROWS = 200
_MAX_TEXT = 160
_SENTIMENT_CLASSES = {"positive", "negative", "neutral", "mixed"}

_STYLE = """
.viz-root {
  color-scheme: light;
  --surface: #fcfcfb; --border: #e4e3de;
  --text-primary: #0b0b0b; --text-secondary: #52514e;
  --positive: #2a78d6; --negative: #e34948; --neutral: #c3c2b7; --bar: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    color-scheme: dark;
    --surface: #1a1a19; --border: #383835;
    --text-primary: #f0efec; --text-secondary: #c3c2b7;
    --positive: #3987e5; --negative: #e66767; --neutral: #6b6a64; --bar: #3987e5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root {
  background: var(--surface); color: var(--text-primary); min-height: 100vh; padding: 24px 16px;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 12px; }
.note { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile, section { border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.tile b { display: block; font-size: 26px; font-weight: 600; }
.tile span { color: var(--text-secondary); font-size: 13px; }
section { margin-top: 16px; }
.row { display: grid; grid-template-columns: minmax(90px, 180px) 1fr 44px; gap: 10px;
  align-items: center; margin: 7px 0; font-size: 14px; }
.name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.value { text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.track { display: flex; gap: 2px; height: 12px; min-width: 0; }
.fill { height: 100%; border-radius: 0 4px 4px 0; min-width: 2px; }
.positive { background: var(--positive); }
.negative { background: var(--negative); }
.neutral, .mixed { background: var(--neutral); }
.bar { background: var(--bar); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px;
  color: var(--text-secondary); margin-bottom: 10px; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.table { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 8px; border-top: 1px solid var(--border);
  vertical-align: top; }
th { color: var(--text-secondary); font-weight: 500; border-top: 0; }
"""


def _tile(value: object, label: str) -> str:
    return f"<div class=tile><b>{escape(str(value))}</b><span>{escape(label)}</span></div>"


def _bars(title: str, items: Sequence[Sequence[Any]], labelled: bool = False) -> str:
    if not items:
        return ""
    top = max(count for _, count in items) or 1
    rows = []
    for name, count in items:
        css = name if labelled and name in _SENTIMENT_CLASSES else "bar"
        safe = escape(str(name), quote=True)
        rows.append(
            f"<div class=row title='{safe}: {count}'><span class=name>{safe}</span>"
            f"<div class=track><div class='fill {css}' style='width:{count / top * 100:.1f}%'>"
            f"</div></div><span class=value>{count}</span></div>"
        )
    return f"<section><h2>{escape(title)}</h2>{''.join(rows)}</section>"


def _aspects(items: Sequence[dict[str, Any]]) -> str:
    if not items:
        return ""
    top = max(item["mentions"] for item in items) or 1
    rows = []
    for item in items:
        name = escape(str(item["name"]), quote=True)
        parts = "".join(
            f"<div class='fill {label}' style='width:{item[label] / top * 100:.1f}%'></div>"
            for label in ("positive", "mixed", "negative")
            if item[label]
        )
        hover = (
            f"{name}: {item['positive']} positive, {item['mixed']} mixed, "
            f"{item['negative']} negative"
        )
        rows.append(
            f"<div class=row title='{hover}'><span class=name>{name}</span>"
            f"<div class=track>{parts}</div><span class=value>{item['mentions']}</span></div>"
        )
    legend = "".join(
        f"<span><i class='key {label}'></i>{label.capitalize()}</span>"
        for label in ("positive", "mixed", "negative")
    )
    return f"<section><h2>Aspects</h2><div class=legend>{legend}</div>{''.join(rows)}</section>"


def _cell(row: dict[str, Any], task: str) -> str:
    if task in row["skipped"]:
        return "skipped"
    if task == "sentiment" and row["sentiment"]:
        return f"{row['sentiment']['label']} ({row['sentiment']['score']:+.2f})"
    if task == "keywords" and row["keywords"]:
        return ", ".join(k["text"] for k in row["keywords"]["keywords"])
    if task == "aspects" and row["aspects"]:
        return ", ".join(f"{a['name']}: {a['label']}" for a in row["aspects"]["aspects"])
    if task == "summarize" and row["summary"]:
        return str(row["summary"]["summary"])
    if task == "stats" and row["stats"]:
        return f"{row['stats']['words']} words"
    if task == "redact" and row["pii"]:
        return f"{len(row['pii']['entities'])} found"
    return ""


def _table(rows: Sequence[dict[str, Any]], tasks: Sequence[str]) -> str:
    if not rows:
        return ""
    heads = "".join(f"<th>{escape(task.capitalize())}</th>" for task in tasks)
    body = []
    for row in rows[:_MAX_ROWS]:
        text = row["text"]
        short = text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "…"
        cells = "".join(f"<td>{escape(_cell(row, task))}</td>" for task in tasks)
        body.append(f"<tr><td>{escape(short)}</td><td>{escape(row['language'])}</td>{cells}</tr>")
    shown = f"Showing the first {_MAX_ROWS} of {len(rows)}." if len(rows) > _MAX_ROWS else ""
    return (
        f"<section><h2>Documents</h2><p class=note>{shown}</p><div class=table><table>"
        f"<tr><th>Text</th><th>Language</th>{heads}</tr>{''.join(body)}</table></div></section>"
    )


def render_html(
    summary: dict[str, Any], rows: Sequence[dict[str, Any]], tasks: Sequence[str]
) -> str:
    """Build the report page."""
    tiles = [_tile(f"{summary['documents']:,}", "Documents")]
    if "average_sentiment" in summary:
        tiles.append(_tile(f"{summary['average_sentiment']:+.2f}", "Average sentiment (-1 to +1)"))
    if "stats" in summary:
        tiles.append(_tile(f"{summary['stats']['total_words']:,}", "Total words"))
        if summary["stats"]["average_reading_ease"] is not None:
            tiles.append(_tile(summary["stats"]["average_reading_ease"], "Average reading ease"))
    if summary.get("pii"):
        tiles.append(_tile(sum(summary["pii"].values()), "Personal details hidden"))
    if summary["skipped"]:
        tiles.append(_tile(summary["skipped"], "Documents with skipped tasks"))

    sections = [
        _bars("Sentiment", list(summary.get("sentiment", {}).items()), labelled=True),
        _aspects(summary.get("aspects", [])),
        _bars("Top keywords", summary.get("keywords", [])),
        _bars("Languages", list(summary["languages"].items())),
        _bars("Personal information found", list(summary.get("pii", {}).items())),
        _table(rows, tasks),
    ]
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>TextInsight report</title><style>{_STYLE}</style></head>"
        "<body><div class=viz-root><main><h1>TextInsight report</h1>"
        "<p class=note>Automatic analysis. Results can be wrong, and sentiment scores are not "
        "calibrated probabilities.</p>"
        f"<div class=tiles>{''.join(tiles)}</div>{''.join(sections)}</main></div></body></html>"
    )
