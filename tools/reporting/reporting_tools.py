from pathlib import Path
from datetime import datetime
import pandas as pd
from fpdf import FPDF

from agent.llm_client import LLMClient

from tools.analysis.analysis_tools import (
    tool_compute_returns,
    get_best_returns_data,
    tool_optimize_markowitz_target
)

from tools.visualization.visualization_tools import (
    plot_sector_allocation,
    plot_sector_performance,
    plot_portfolio_value_over_time,
    plot_portfolio_performance,
    plot_allocation_vs_markowitz,
    plot_asset_correlation_heatmap,
    plot_portfolio_vs_benchmark
)

# ============================
# PATHS
# ============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = PROJECT_ROOT / "plots"

REPORTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================
# HELPERS
# ============================

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

import re

def _unique_pdf_path(default_prefix: str, output_pdf: str | None, reports_dir: Path) -> str:
    """
    Build a unique PDF output path to avoid overwriting existing reports.
    If output_pdf is:
      - None/empty -> reports_dir/{default_prefix}_{timestamp}.pdf
      - a directory -> create timestamped file inside it
      - a filename -> append timestamp unless already timestamped
    """
    ts = _timestamp()

    def looks_timestamped(stem: str) -> bool:
        return re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", stem) is not None

    if not output_pdf:
        return str(reports_dir / f"{default_prefix}_{ts}.pdf")

    out = Path(output_pdf)

    # Directory path (no suffix)
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return str(out / f"{default_prefix}_{ts}.pdf")

    # File path
    if looks_timestamped(out.stem):
        return str(out)

    return str(out.with_name(f"{out.stem}_{ts}{out.suffix}"))


def _fmt_money(value):
    """Format as 123'456.78 (apostrophe thousands separator)."""
    try:
        x = float(value)
        s = f"{x:,.2f}"
        return s.replace(",", "'")
    except Exception:
        return str(value)


def _safe_text(text: str, max_len=22):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def _ensure_space(pdf: FPDF, needed_height=70):
    """If there isn't enough space on the current page, go to a new page."""
    if pdf.get_y() + needed_height > 270:
        pdf.add_page()


def _extract_date_range(df) -> tuple[str | None, str | None]:
    """
    Robust extraction of min/max dates from:
      - date-like column names (date/time/timestamp/datetime)
      - or datetime index
    """
    if df is None or getattr(df, "empty", True):
        return None, None

    try:
        # 1) search for a date-like column
        candidate_cols = []
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ["date", "time", "timestamp", "datetime"]):
                candidate_cols.append(c)

        if candidate_cols:
            s = pd.to_datetime(df[candidate_cols[0]], errors="coerce")
        else:
            s = pd.to_datetime(df.index, errors="coerce")

        s = s.dropna()
        if s.empty:
            return None, None

        return s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d")
    except Exception:
        return None, None


def _clean_best_returns_table(rows: list) -> pd.DataFrame:
    """
    Clean best_returns table for PDF readability.
    Keeps key columns and renames them.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    rename_map = {
        "ticker": "Ticker",
        "sector": "Sector",
        "quantity": "Qty",
        "avg_purchase_price": "Avg Buy",
        "current_market_price": "Price",
        "invested_cost": "Cost",
        "current_value": "Value",
        "profit_loss": "P/L",
        "return_percentage": "Return %"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    preferred = ["Ticker", "Qty", "Avg Buy", "Price", "Cost", "Value", "P/L", "Return %", "Sector"]
    df = df[[c for c in preferred if c in df.columns]]

    for col in ["Avg Buy", "Price", "Cost", "Value", "P/L"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _fmt_money(x) if isinstance(x, (int, float)) else x)

    if "Return %" in df.columns:
        df["Return %"] = df["Return %"].apply(lambda x: f"{float(x):.2f}%" if isinstance(x, (int, float)) else x)

    return df


def _build_ai_prompt(roi: dict, df_portfolio: pd.DataFrame, start_date: str | None, end_date: str | None) -> list[dict]:
    """
    Build messages for AI commentary (English, professional).
    """
    # Extract a few key points
    period_str = f"{start_date} → {end_date}" if start_date and end_date else "N/A"

    roi_pct = roi.get("roi_percentage", "N/A")
    invested = roi.get("total_invested_cost", "N/A")
    current = roi.get("total_current_value", "N/A")
    currency = roi.get("currency", "")

    # Try to compute top/bottom performers from df_portfolio (if Return % exists)
    top3, bottom3 = "", ""
    if not df_portfolio.empty and "Return %" in df_portfolio.columns:
        # Convert "12.34%" -> float
        tmp = df_portfolio.copy()
        tmp["_ret"] = tmp["Return %"].astype(str).str.replace("%", "", regex=False)
        tmp["_ret"] = pd.to_numeric(tmp["_ret"], errors="coerce")

        top = tmp.sort_values("_ret", ascending=False).head(3)
        bottom = tmp.sort_values("_ret", ascending=True).head(3)

        def fmt_rows(d):
            out = []
            for _, r in d.iterrows():
                out.append(f"{r.get('Ticker','?')} ({r.get('Return %','?')})")
            return ", ".join(out)

        top3 = fmt_rows(top)
        bottom3 = fmt_rows(bottom)

    data_summary = f"""
PERIOD: {period_str}
ROI: {roi_pct}%
INVESTED: {invested} {currency}
CURRENT: {current} {currency}
TOP 3 PERFORMERS: {top3 or "N/A"}
BOTTOM 3 PERFORMERS: {bottom3 or "N/A"}
COLUMNS AVAILABLE: {", ".join(df_portfolio.columns.tolist()) if not df_portfolio.empty else "N/A"}
""".strip()

    system = (
        "You are a professional AI portfolio advisor. "
        "Write concise, data-grounded insights. "
        "Do not include legal/medical disclaimers. Do not mention tool calls."
    )

    user = (
        "Generate an 'AI Insights & Recommendations' section based only on the provided portfolio summary.\n\n"
        f"{data_summary}\n\n"
        "Output format (plain text, NO markdown):\n"
        "INSIGHTS:\n"
        "- <bullet 1>\n"
        "- <bullet 2>\n"
        "- <bullet 3>\n"
        "- <bullet 4>\n"
        "- <bullet 5>\n"
        "RECOMMENDATIONS:\n"
        "- <bullet 1>\n"
        "- <bullet 2>\n"
        "RISKS TO WATCH:\n"
        "- <bullet 1>\n"
        "Rules:\n"
        "- Do NOT use markdown symbols like **, *, or backticks.\n"
        "- Keep it brief, specific, and grounded in the provided numbers.\n"
        "- Keep each bullet to max 1 sentence.\n"
        "- Always reference ROI and mention at least one top performer and one bottom performer.\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _sanitize_for_pdf(text: str) -> str:
    if not text:
        return ""

    text = text.replace("**", "")
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("* "):
            # convert "* foo" -> "foo"
            line = line.replace("* ", "", 1)
        lines.append(line)

    return "\n".join(lines).strip()

# ============================
# PDF CLASS
# ============================

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 13)
        self.cell(0, 8, "AI Portfolio Advisor - Report", ln=True, align="C")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def _section_title(pdf: FPDF, title: str):
    _ensure_space(pdf, 22)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, title, ln=True)
    pdf.ln(2)


def _draw_table(pdf: FPDF, df: pd.DataFrame, font_size=8, row_h=6, max_rows=14):
    if df is None or df.empty:
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, "No data available.", ln=True)
        pdf.ln(4)
        return

    df = df.head(max_rows).copy()
    headers = list(df.columns)
    n = len(headers)

    page_width = 190
    col_w = int(page_width / n)
    col_widths = [col_w] * n

    def draw_header():
        pdf.set_font("Arial", "B", font_size)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], row_h, str(h), border=1, align="C")
        pdf.ln()

    _ensure_space(pdf, 50)
    draw_header()
    pdf.set_font("Arial", "", font_size)

    for _, row in df.iterrows():
        if pdf.get_y() > 265:
            pdf.add_page()
            draw_header()
            pdf.set_font("Arial", "", font_size)

        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], row_h, _safe_text(row[h], 22), border=1)
        pdf.ln()

    pdf.ln(6)


def _add_image(pdf: FPDF, title: str, img_path: Path | None, width=160):
    _ensure_space(pdf, 95)  # ensure title+image together
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, title, ln=True)
    pdf.ln(2)

    if img_path and img_path.exists():
        pdf.image(str(img_path), w=width)
    else:
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"Image not found: {img_path}", ln=True)

    pdf.ln(8)


# ==========================================================
# REPORT TOOL ON PORTFOLIO PERFORMANCE (PDF ONLY)
# ==========================================================

def generate_portfolio_report(output_pdf: str | None = None, include_ai_commentary: bool = True) -> dict:
    """
    PDF-only report with this structure:

    1) Current portfolio table + plot_portfolio_performance
    2) Portfolio value over time plot + tool_compute_returns (+ period analyzed)
    3) Sector allocation + plot_sector_performance
    4) (Optional) AI Insights & Recommendations (Groq text)
    """

    # Always produce a unique filename unless user explicitly provided a timestamped name
    output_pdf = _unique_pdf_path("portfolio_overview", output_pdf, REPORTS_DIR)


    # ===== Data =====
    roi = tool_compute_returns()
    best_returns = get_best_returns_data()
    df_portfolio = _clean_best_returns_table(best_returns)

    # ===== Plots (use returned image_path due to timestamps) =====
    perf_out = plot_portfolio_performance(save_path=str(PLOTS_DIR / "portfolio_performance.png"))
    value_out = plot_portfolio_value_over_time(save_path=str(PLOTS_DIR / "portfolio_value_over_time.png"))
    sector_alloc_out = plot_sector_allocation(save_path=str(PLOTS_DIR / "sector_allocation.png"))
    sector_perf_out = plot_sector_performance(save_path=str(PLOTS_DIR / "sector_performance.png"))

    perf_path = Path(perf_out["image_path"]) if perf_out.get("image_path") else None
    value_path = Path(value_out["image_path"]) if value_out.get("image_path") else None
    sector_alloc_path = Path(sector_alloc_out["image_path"]) if sector_alloc_out.get("image_path") else None
    sector_perf_path = Path(sector_perf_out["image_path"]) if sector_perf_out.get("image_path") else None

    # Extract analysis period from value_over_time data
    start_date, end_date = _extract_date_range(value_out.get("data"))

    # ===== Build PDF =====
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Portfolio Overview Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    # ============================
    # SECTION 1: Current Portfolio
    # ============================
    _section_title(pdf, "1) Current Portfolio Overview")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Current holdings summary with key performance metrics.")
    pdf.ln(2)

    _draw_table(pdf, df_portfolio, font_size=8, row_h=6, max_rows=14)
    _add_image(pdf, "Portfolio Performance", perf_path, width=160)

    # ============================
    # SECTION 2: Value Over Time + ROI
    # ============================
    _section_title(pdf, "2) Portfolio Value Over Time + ROI Summary")
    pdf.set_font("Arial", "", 11)

    if start_date and end_date:
        pdf.multi_cell(0, 6, f"Period analyzed: {start_date} -> {end_date}")
        pdf.ln(1)

    if "error" in roi:
        pdf.multi_cell(0, 6, f"ROI could not be computed: {roi['error']}")
    else:
        invested = _fmt_money(roi.get("total_invested_cost"))
        current = _fmt_money(roi.get("total_current_value"))
        roi_pct = roi.get("roi_percentage")
        currency = roi.get("currency", "")

        pdf.multi_cell(
            0, 6,
            f"ROI: {roi_pct}% | Invested: {invested} {currency} | Current: {current} {currency}"
        )

    pdf.ln(3)
    _add_image(pdf, "Portfolio Value Over Time", value_path, width=160)

    # ============================
    # SECTION 3: Sector Insights
    # ============================
    _section_title(pdf, "3) Sector Allocation & Sector Performance")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Diversification insights based on sector allocation and sector performance.")
    pdf.ln(2)

    _add_image(pdf, "Sector Allocation", sector_alloc_path, width=155)
    _add_image(pdf, "Sector Performance", sector_perf_path, width=155)

    # ============================
    # SECTION 4: AI Insights (optional)
    # ============================
    if include_ai_commentary:
        _section_title(pdf, "4) AI Insights & Recommendations")

        llm = LLMClient()  
        messages = _build_ai_prompt(roi, df_portfolio, start_date, end_date)
        commentary = llm.chat_text(messages=messages, temperature=0.0, max_tokens=450)
        commentary = _sanitize_for_pdf(commentary)

        pdf.set_font("Arial", "", 11)
        # Ensure there is space for a few lines; otherwise move page
        _ensure_space(pdf, 90)
        pdf.multi_cell(0, 6, commentary)
        pdf.ln(4)

    # ===== Save =====
    try:
        pdf.output(output_pdf)
        return {
            "status": "ok",
            "format": "pdf",
            "report_path": output_pdf,
            "message": "Portfolio overview PDF report generated successfully."
        }
    except Exception as e:
        return {
            "status": "error",
            "format": "pdf",
            "report_path": output_pdf,
            "message": f"PDF generation failed: {e}"
        }



# ==========================================================
# REPORT ON RISK OPTIMIZATION TOOL (PDF ONLY)
# ==========================================================

def generate_risk_optimization_report(
    target_return: float = 0.10,
    start_date: str = "2025-01-02",
    end_date: str | None = None,
    output_pdf: str | None = None,
    include_ai_commentary: bool = True
) -> dict:
    """
    Risk & Optimization PDF report:
    - Asset correlation heatmap (risk/diversification)
    - Markowitz optimization for a target return (table from tool OR fallback from plot data)
    - Allocation vs Markowitz plot (and its dataframe)
    - Portfolio vs benchmark plot
    - Optional AI insights (Groq text)
    """

    # Unique output path (avoid overwrite)
    output_pdf = _unique_pdf_path("risk_optimization", output_pdf, REPORTS_DIR)

    # ---------- Markowitz optimization (tool) ----------
    opt = tool_optimize_markowitz_target(target_return_annualized=target_return)

    # Try to get weights from tool output (may be empty depending on implementation)
    weights = (
        opt.get("weights")
        or opt.get("optimal_weights")
        or opt.get("markowitz_weights")
        or {}
    )

    # Build weights table from tool output (Ticker + Weight fraction or pct unknown)
    df_w = pd.DataFrame([{"Ticker": k, "Weight": v} for k, v in weights.items()])
    if not df_w.empty:
        df_w["Weight"] = pd.to_numeric(df_w["Weight"], errors="coerce")
        df_w = df_w.sort_values("Weight", ascending=False)
        # If weights look like fractions (0-1), print as %, else still ok
        # Here we just format as % if <= 1.5 on average; otherwise keep as % anyway.
        avg_w = df_w["Weight"].dropna().mean() if not df_w["Weight"].dropna().empty else None
        if avg_w is not None and avg_w <= 1.5:
            df_w["Weight"] = df_w["Weight"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        else:
            df_w["Weight"] = df_w["Weight"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    # Extract any metrics if available
    metric_lines = []
    for key, label, fmt in [
        ("expected_return", "Expected return", "{:.2%}"),
        ("volatility", "Volatility", "{:.2%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("sharpe_ratio", "Sharpe", "{:.2f}"),
    ]:
        if isinstance(opt, dict) and key in opt and opt[key] is not None:
            try:
                metric_lines.append(f"{label}: {fmt.format(float(opt[key]))}")
            except Exception:
                metric_lines.append(f"{label}: {opt[key]}")

    # ---------- Plots (IMPORTANT: use returned image_path because timestamped) ----------
    heat_out = plot_asset_correlation_heatmap(
        start_date=start_date,
        end_date=end_date,
        save_path=str(PLOTS_DIR / "asset_correlation_heatmap.png")
    )

    alloc_out = plot_allocation_vs_markowitz(
        target_return=target_return,
        save_path=str(PLOTS_DIR / "allocation_vs_markowitz.png")
    )

    bench_out = plot_portfolio_vs_benchmark(
        benchmark_ticker="^GSPC",
        start_date=start_date,
        end_date=end_date,
        save_path=str(PLOTS_DIR / "portfolio_vs_benchmark.png")
    )

    heat_path = Path(heat_out["image_path"]) if heat_out.get("image_path") else None
    alloc_path = Path(alloc_out["image_path"]) if alloc_out.get("image_path") else None
    bench_path = Path(bench_out["image_path"]) if bench_out.get("image_path") else None

    # ---------- Fallback: build weights table from alloc_out['data'] ----------
    # alloc_out returns: df[['ticker','current_weight_pct','markowitz_weight_pct']]
    alloc_df = alloc_out.get("data")

    if (df_w is None or df_w.empty) and alloc_df is not None and not getattr(alloc_df, "empty", True):
        df_w = alloc_df.copy()

        # Normalize column names
        ren = {
            "ticker": "Ticker",
            "current_weight_pct": "Current Weight",
            "markowitz_weight_pct": "Markowitz Weight"
        }
        df_w = df_w.rename(columns={k: v for k, v in ren.items() if k in df_w.columns})

        # Format percent columns (your data is 0-100)
        for c in ["Current Weight", "Markowitz Weight"]:
            if c in df_w.columns:
                df_w[c] = pd.to_numeric(df_w[c], errors="coerce")
                df_w[c] = df_w[c].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

        # Sort by Markowitz weight desc (using original numeric column)
        try:
            tmp = alloc_df.copy()
            tmp["markowitz_weight_pct"] = pd.to_numeric(tmp["markowitz_weight_pct"], errors="coerce")
            order = tmp.sort_values("markowitz_weight_pct", ascending=False)["ticker"].tolist()
            df_w["__order"] = df_w["Ticker"].apply(lambda t: order.index(t) if t in order else 9999)
            df_w = df_w.sort_values("__order").drop(columns=["__order"])
        except Exception:
            pass

    # ---------- Build PDF ----------
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Risk & Optimization Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 11)
    # ASCII arrow to avoid unicode font issues
    pdf.multi_cell(0, 6, f"Analysis window: {start_date} -> {end_date or 'latest'} | Markowitz target return: {target_return:.0%}")
    pdf.ln(6)

    # 1) Correlation heatmap
    _section_title(pdf, "1) Diversification Risk: Asset Correlation")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Higher correlations indicate lower diversification benefits across holdings.")
    pdf.ln(2)
    _add_image(pdf, "Asset Correlation Heatmap", heat_path, width=160)

    # 2) Markowitz weights
    _section_title(pdf, "2) Markowitz Optimization (Target Return)")
    pdf.set_font("Arial", "", 11)
    if metric_lines:
        pdf.multi_cell(0, 6, " | ".join(metric_lines))
        pdf.ln(2)

    _section_title(pdf, "Optimized Weights (Top)")
    _draw_table(pdf, df_w.head(12) if df_w is not None else pd.DataFrame(), font_size=9, row_h=7, max_rows=12)

    # 3) Allocation comparison
    _section_title(pdf, "3) Current vs Optimized Allocation")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Comparison between current portfolio weights and Markowitz optimized weights.")
    pdf.ln(2)
    _add_image(pdf, "Allocation vs Markowitz", alloc_path, width=160)

    # 4) Benchmark comparison
    _section_title(pdf, "4) Portfolio vs Benchmark")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Performance comparison against a benchmark (default: S&P 500).")
    pdf.ln(2)
    _add_image(pdf, "Portfolio vs Benchmark (^GSPC)", bench_path, width=160)

    # 5) AI commentary
    if include_ai_commentary:
        _section_title(pdf, "5) AI Insights (Risk & Optimization)")
        llm = LLMClient()

        # Build facts for commentary
        top_tickers = ", ".join(df_w["Ticker"].head(5).tolist()) if df_w is not None and not df_w.empty and "Ticker" in df_w.columns else "N/A"
        facts = (
            f"Window: {start_date} to {end_date or 'latest'}\n"
            f"Target return: {target_return:.0%}\n"
            f"Markowitz metrics: {(' | '.join(metric_lines) if metric_lines else 'N/A')}\n"
            f"Top optimized tickers: {top_tickers}\n"
            "Note: Weights shown are % allocations.\n"
        )

        prompt = (
            "Write a concise risk & optimization commentary in English (NO markdown).\n"
            "Structure exactly:\n"
            "INSIGHTS:\n- (3 bullets)\n"
            "RECOMMENDATIONS:\n- (2 bullets)\n"
            "RISKS TO WATCH:\n- (1 bullet)\n"
            "Rules: max 1 sentence per bullet. Use only the facts provided.\n\n"
            f"FACTS:\n{facts}"
        )

        txt = llm.chat_text(
            messages=[
                {"role": "system", "content": "You are a professional AI portfolio advisor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=320
        )
        txt = _sanitize_for_pdf(txt) if "_sanitize_for_pdf" in globals() else txt

        _ensure_space(pdf, 95)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, txt)
        pdf.ln(4)

    # Save
    try:
        pdf.output(output_pdf)
        return {"status": "ok", "format": "pdf", "report_path": output_pdf, "message": "Risk & optimization report generated."}
    except Exception as e:
        return {"status": "error", "format": "pdf", "report_path": output_pdf, "message": f"PDF generation failed: {e}"}

