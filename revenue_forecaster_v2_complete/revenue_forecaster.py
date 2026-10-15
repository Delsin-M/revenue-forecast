import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

st.set_page_config(page_title="Revenue Forecasting Simulator V2", page_icon="📈", layout="wide")

DB_PATH = Path("forecasts.db")
STAGES = {
    "Booked": 1.00,
    "White Paper": 0.05,
    "Invited Proposal": 0.15,
    "Proposal Submitted": 0.30,
    "Competitive Range": 0.55,
    "Final Negotiation": 0.85,
    "Awaiting Award": 0.95,
}
COLUMNS = [
    "Program", "Status", "Opportunity Stage", "Amount", "Start Date", "End Date",
    "Win Probability", "Use Stage Probability", "Funding Agency", "Capture Manager",
    "Prime/Sub", "Prime Organization"
]
DEFAULT = pd.DataFrame([
    ["Booked Program Alpha", "Booked", "Booked", 1200000, "2026-01-01", "2026-12-31", 1.00, True, "DARPA", "A. Manager", "Prime", ""],
    ["Bid Program Beta", "Bid", "Proposal Submitted", 900000, "2026-04-01", "2027-03-31", 0.30, True, "AFRL", "B. Manager", "Sub", "Example Prime"],
    ["Bid Program Gamma", "Bid", "Competitive Range", 600000, "2026-07-01", "2027-06-30", 0.55, True, "ONR", "C. Manager", "Prime", ""],
], columns=COLUMNS)
DEFAULT["Start Date"] = pd.to_datetime(DEFAULT["Start Date"])
DEFAULT["End Date"] = pd.to_datetime(DEFAULT["End Date"])
DEFAULT["Use Stage Probability"] = DEFAULT["Use Stage Probability"].astype(bool)


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS forecasts (
        name TEXT PRIMARY KEY, description TEXT, created_at TEXT, updated_at TEXT,
        programs_json TEXT NOT NULL, settings_json TEXT NOT NULL)""")
    con.commit()
    return con


def db_list():
    with db() as con:
        return pd.read_sql_query("SELECT name, description, updated_at FROM forecasts ORDER BY updated_at DESC", con)


def save_forecast(name, description, programs, settings):
    payload = programs.copy()
    payload["Start Date"] = pd.to_datetime(payload["Start Date"]).dt.strftime("%Y-%m-%d")
    payload["End Date"] = pd.to_datetime(payload["End Date"]).dt.strftime("%Y-%m-%d")
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with db() as con:
        old = con.execute("SELECT created_at FROM forecasts WHERE name=?", (name,)).fetchone()
        created = old[0] if old else now
        con.execute("""INSERT OR REPLACE INTO forecasts
            (name, description, created_at, updated_at, programs_json, settings_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, created, now, payload.to_json(orient="records"), json.dumps(settings)))
        con.commit()


def load_forecast(name):
    with db() as con:
        row = con.execute("SELECT description, programs_json, settings_json FROM forecasts WHERE name=?", (name,)).fetchone()
    if not row:
        raise ValueError("Forecast not found.")
    frame = pd.read_json(io.StringIO(row[1]), orient="records")
    frame["Start Date"] = pd.to_datetime(frame["Start Date"])
    frame["End Date"] = pd.to_datetime(frame["End Date"])
    frame["Use Stage Probability"] = frame["Use Stage Probability"].astype(bool)
    return row[0], frame[COLUMNS], json.loads(row[2])


def delete_forecast(name):
    with db() as con:
        con.execute("DELETE FROM forecasts WHERE name=?", (name,)); con.commit()


def normalized(df):
    d = df.copy()
    for c in COLUMNS:
        if c not in d.columns:
            if c == "Use Stage Probability": d[c] = True
            elif c in ["Capture Manager", "Prime Organization"]: d[c] = ""
            elif c == "Prime/Sub": d[c] = "Prime"
            elif c == "Opportunity Stage": d[c] = "Proposal Submitted"
            else: raise ValueError(f"Missing required column: {c}")
    d = d.dropna(subset=["Program", "Status", "Amount", "Start Date", "End Date"])
    d["Program"] = d["Program"].astype(str).str.strip()
    d["Status"] = d["Status"].astype(str).str.strip().str.title()
    d["Opportunity Stage"] = d["Opportunity Stage"].fillna("Proposal Submitted").astype(str)
    d["Amount"] = pd.to_numeric(d["Amount"], errors="coerce")
    d["Start Date"] = pd.to_datetime(d["Start Date"], errors="coerce")
    d["End Date"] = pd.to_datetime(d["End Date"], errors="coerce")
    d["Win Probability"] = pd.to_numeric(d["Win Probability"], errors="coerce").fillna(0).clip(0, 1)
    d["Use Stage Probability"] = d["Use Stage Probability"].fillna(True).astype(bool)
    for c in ["Funding Agency", "Capture Manager", "Prime/Sub", "Prime Organization"]:
        d[c] = d[c].fillna("Unspecified").astype(str)
    if d.empty: raise ValueError("Add at least one complete program.")
    if d[["Amount", "Start Date", "End Date"]].isna().any().any(): raise ValueError("Amounts and dates must be valid.")
    if (~d["Status"].isin(["Booked", "Bid"])).any(): raise ValueError("Status must be Booked or Bid.")
    if (d["Amount"] < 0).any(): raise ValueError("Amounts cannot be negative.")
    if (d["End Date"] < d["Start Date"]).any(): raise ValueError("Each end date must be on or after its start date.")
    d["Effective Probability"] = d.apply(lambda r: 1.0 if r["Status"] == "Booked" else (STAGES.get(r["Opportunity Stage"], r["Win Probability"]) if r["Use Stage Probability"] else r["Win Probability"]), axis=1)
    d.loc[d["Status"].eq("Booked"), "Opportunity Stage"] = "Booked"
    return d.reset_index(drop=True)


def allocations(d):
    first, last = d["Start Date"].dt.to_period("M").min(), d["End Date"].dt.to_period("M").max()
    months = pd.period_range(first, last, freq="M")
    matrix = np.zeros((len(d), len(months)))
    for i, r in d.iterrows():
        mask = (months >= r["Start Date"].to_period("M")) & (months <= r["End Date"].to_period("M"))
        matrix[i, mask] = r["Amount"] / int(mask.sum())
    return months, matrix


def simulate(d, runs, seed):
    months, alloc = allocations(d)
    booked = d["Status"].eq("Booked").to_numpy()
    bids = np.where(d["Status"].eq("Bid").to_numpy())[0]
    rng = np.random.default_rng(int(seed))
    inclusion = np.tile(booked, (int(runs), 1))
    if len(bids):
        inclusion[:, bids] = rng.random((int(runs), len(bids))) < d.loc[bids, "Effective Probability"].to_numpy()
    revenues = inclusion.astype(float) @ alloc
    totals = revenues.sum(axis=1)
    if len(bids):
        keys = [row.tobytes() for row in np.packbits(inclusion[:, bids], axis=1)]
        counts = pd.Series(keys).value_counts()
        mode_key, mode_count = counts.index[0], int(counts.iloc[0])
        mode_run = keys.index(mode_key)
    else:
        mode_count, mode_run = int(runs), 0
    monthly = pd.DataFrame({
        "Month": months.to_timestamp(),
        "Most Likely Specific Scenario": revenues[mode_run],
        "Average Scenario": revenues.mean(axis=0),
        "P10": np.percentile(revenues, 10, axis=0),
        "P50": np.percentile(revenues, 50, axis=0),
        "P90": np.percentile(revenues, 90, axis=0),
    })
    modal_wins = d.loc[inclusion[mode_run] & d["Status"].eq("Bid").to_numpy(), "Program"].tolist()
    sensitivity = d.loc[d["Status"].eq("Bid"), ["Program", "Amount", "Effective Probability", "Funding Agency"]].copy()
    sensitivity["Revenue Swing"] = sensitivity["Amount"]
    sensitivity["Expected Revenue"] = sensitivity["Amount"] * sensitivity["Effective Probability"]
    sensitivity = sensitivity.sort_values("Revenue Swing", ascending=False)
    return {"monthly": monthly, "totals": totals, "inclusion": inclusion, "mode_frequency": mode_count / int(runs), "modal_wins": modal_wins, "alloc": alloc, "months": months, "sensitivity": sensitivity}


def fiscal_summary(monthly, fy_start):
    x = monthly.copy()
    x["Fiscal Year"] = x["Month"].apply(lambda z: z.year + 1 if z.month >= fy_start and fy_start != 1 else z.year)
    return x.groupby("Fiscal Year", as_index=False)[["Most Likely Specific Scenario", "Average Scenario", "P10", "P50", "P90"]].sum()


def pdf_report(name, description, d, result, fy):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet(); story = []
    story += [Paragraph(f"Revenue Forecast Executive Report: {name}", styles["Title"]), Paragraph(description or "", styles["BodyText"]), Spacer(1, 12)]
    totals = result["totals"]
    metrics = [["Metric", "Value"], ["Average total", f"${totals.mean():,.0f}"], ["P10 total", f"${np.percentile(totals,10):,.0f}"], ["Median total", f"${np.percentile(totals,50):,.0f}"], ["P90 total", f"${np.percentile(totals,90):,.0f}"], ["Most likely scenario frequency", f"{result['mode_frequency']:.1%}"]]
    mt = Table(metrics, colWidths=[2.5*inch, 2.2*inch]); mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0f766e")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"RIGHT"),("PADDING",(0,0),(-1,-1),6)])); story += [mt, Spacer(1, 16)]
    story += [Paragraph("Most likely bid wins", styles["Heading2"]), Paragraph(", ".join(result["modal_wins"]) if result["modal_wins"] else "No bid programs won in the modal scenario.", styles["BodyText"]), Spacer(1, 12)]
    fs = fiscal_summary(result["monthly"], fy)
    data = [["FY", "Most Likely", "Average", "P10", "P50", "P90"]] + [[str(int(r["Fiscal Year"]))] + [f"${r[c]:,.0f}" for c in ["Most Likely Specific Scenario","Average Scenario","P10","P50","P90"]] for _,r in fs.iterrows()]
    ft = Table(data, repeatRows=1); ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1d4ed8")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.5,colors.grey),("ALIGN",(1,1),(-1,-1),"RIGHT"),("PADDING",(0,0),(-1,-1),5)])); story += [Paragraph("Fiscal-year forecast", styles["Heading2"]), ft, PageBreak()]
    pdata = [["Program","Status","Stage","Agency","Amount","P(win)"]] + [[str(r["Program"]),str(r["Status"]),str(r["Opportunity Stage"]),str(r["Funding Agency"]),f"${r['Amount']:,.0f}",f"{r['Effective Probability']:.0%}"] for _,r in d.iterrows()]
    pt = Table(pdata, repeatRows=1, colWidths=[2.1*inch,.7*inch,1.4*inch,1.0*inch,1.0*inch,.7*inch]); pt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#334155")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("ALIGN",(4,1),(-1,-1),"RIGHT")]))
    story += [Paragraph("Program portfolio", styles["Heading2"]), pt]
    doc.build(story); buf.seek(0); return buf.getvalue()

st.title("Revenue Forecasting Simulator V2")
st.caption("Saved scenarios, stage-based win probabilities, Monte Carlo forecasting, executive summaries, and report exports.")

if "programs" not in st.session_state: st.session_state.programs = DEFAULT.copy()
if "forecast_name" not in st.session_state: st.session_state.forecast_name = "Baseline Forecast"
if "description" not in st.session_state: st.session_state.description = ""
if "runs" not in st.session_state: st.session_state.runs = 10000
if "seed" not in st.session_state: st.session_state.seed = 42
if "fy_start" not in st.session_state: st.session_state.fy_start = 10

with st.sidebar:
    st.header("Forecast library")
    saved = db_list()
    selected = st.selectbox("Saved forecast", [""] + saved["name"].tolist())
    c1,c2 = st.columns(2)
    if c1.button("Load", width="stretch", disabled=not selected):
        desc, frame, settings = load_forecast(selected)
        st.session_state.programs = frame; st.session_state.forecast_name = selected; st.session_state.description = desc
        st.session_state.runs = int(settings.get("runs",10000)); st.session_state.seed = int(settings.get("seed",42)); st.session_state.fy_start = int(settings.get("fy_start",10)); st.session_state.pop("result",None); st.rerun()
    if c2.button("Delete", width="stretch", disabled=not selected): delete_forecast(selected); st.rerun()
    st.divider(); st.header("Simulation")
    st.session_state.runs = st.number_input("Monte Carlo runs", 100, 250000, st.session_state.runs, 1000)
    st.session_state.seed = st.number_input("Random seed", 0, value=st.session_state.seed, step=1)
    st.session_state.fy_start = st.selectbox("Fiscal year begins", list(range(1,13)), index=st.session_state.fy_start-1, format_func=lambda m: pd.Timestamp(2026,m,1).strftime("%B"))
    upload = st.file_uploader("Import CSV or XLSX", type=["csv","xlsx"])
    if upload:
        try:
            frame = pd.read_excel(upload, engine="openpyxl") if upload.name.lower().endswith("xlsx") else pd.read_csv(upload)
            frame["Start Date"] = pd.to_datetime(frame["Start Date"]); frame["End Date"] = pd.to_datetime(frame["End Date"])
            for col in COLUMNS:
                if col not in frame.columns:
                    if col == "Use Stage Probability": frame[col] = True
                    elif col == "Opportunity Stage": frame[col] = "Proposal Submitted"
                    else: frame[col] = ""
            st.session_state.programs = frame[COLUMNS]; st.session_state.pop("result",None); st.success("Imported. Review inputs, then run simulation.")
        except Exception as e: st.error(f"Import failed: {e}")

st.session_state.forecast_name = st.text_input("Forecast name", value=st.session_state.forecast_name)
st.session_state.description = st.text_area("Description / assumptions", value=st.session_state.description, height=80)

edited = st.data_editor(st.session_state.programs, num_rows="dynamic", width="stretch", hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn(options=["Booked","Bid"], required=True),
        "Opportunity Stage": st.column_config.SelectboxColumn(options=list(STAGES), required=True),
        "Amount": st.column_config.NumberColumn(min_value=0.0, format="$%.2f", required=True),
        "Start Date": st.column_config.DateColumn(format="YYYY-MM-DD", required=True),
        "End Date": st.column_config.DateColumn(format="YYYY-MM-DD", required=True),
        "Win Probability": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=.01, format="%.2f"),
        "Use Stage Probability": st.column_config.CheckboxColumn(help="When checked, the opportunity stage determines P(win)."),
        "Prime/Sub": st.column_config.SelectboxColumn(options=["Prime","Sub"]),
    }, key="program_editor")
st.session_state.programs = edited

b1,b2,b3 = st.columns([1,1,2])
if b1.button("Save forecast", type="secondary", width="stretch"):
    try:
        d = normalized(edited); settings={"runs":int(st.session_state.runs),"seed":int(st.session_state.seed),"fy_start":int(st.session_state.fy_start)}
        save_forecast(st.session_state.forecast_name.strip(), st.session_state.description, d[COLUMNS], settings); st.success("Forecast saved to this app instance.")
    except Exception as e: st.error(str(e))
if b2.button("Run simulation", type="primary", width="stretch"):
    try:
        d=normalized(edited); st.session_state.result=(d,simulate(d,st.session_state.runs,st.session_state.seed)); st.rerun()
    except Exception as e: st.error(str(e))

if "result" in st.session_state:
    d,r = st.session_state.result; totals=r["totals"]; monthly=r["monthly"]
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Booked",f"${d.loc[d.Status.eq('Booked'),'Amount'].sum():,.0f}")
    m2.metric("Average total",f"${totals.mean():,.0f}")
    m3.metric("Median",f"${np.percentile(totals,50):,.0f}")
    m4.metric("P10",f"${np.percentile(totals,10):,.0f}")
    m5.metric("P90",f"${np.percentile(totals,90):,.0f}")
    tabs=st.tabs(["Executive dashboard","Monthly forecast","Fiscal years","Agencies & managers","Sensitivity","Scenario detail","Exports"])
    with tabs[0]:
        left,right=st.columns([2,1])
        with left:
            fig=go.Figure(); fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.P90,line=dict(width=0),showlegend=False)); fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.P10,fill="tonexty",fillcolor="rgba(14,165,233,.15)",line=dict(width=0),name="P10–P90")); fig.add_trace(go.Scatter(x=monthly.Month,y=monthly["Average Scenario"],name="Average",line=dict(color="#10b981",width=3))); fig.add_trace(go.Scatter(x=monthly.Month,y=monthly["Most Likely Specific Scenario"],name="Most likely scenario",line=dict(color="#0ea5e9",width=3))); fig.update_layout(height=430,hovermode="x unified",yaxis_title="Revenue",legend_orientation="h",template="plotly_white"); st.plotly_chart(fig,width="stretch")
        with right:
            st.subheader("Executive highlights"); st.write(f"**Most likely scenario frequency:** {r['mode_frequency']:.1%}"); st.write("**Most likely bid wins:** " + (", ".join(r["modal_wins"]) if r["modal_wins"] else "None")); top=r["sensitivity"].head(3); st.write("**Largest opportunity swings:**"); [st.write(f"• {x.Program}: ${x['Revenue Swing']:,.0f}") for _,x in top.iterrows()]
    with tabs[1]:
        show=monthly.copy(); show.Month=show.Month.dt.strftime("%Y-%m"); st.dataframe(show.style.format({c:"${:,.0f}" for c in show.columns if c!="Month"}),width="stretch",hide_index=True)
    with tabs[2]:
        fy=fiscal_summary(monthly,st.session_state.fy_start); st.dataframe(fy.style.format({c:"${:,.0f}" for c in fy.columns if c!="Fiscal Year"}),width="stretch",hide_index=True)
    with tabs[3]:
        agency=d.groupby("Funding Agency",as_index=False).agg(Booked=("Amount",lambda s:s[d.loc[s.index,"Status"].eq("Booked")].sum()),Expected=("Amount",lambda s:(s*d.loc[s.index,"Effective Probability"]).sum()))
        manager=d.groupby("Capture Manager",as_index=False).apply(lambda g:pd.Series({"Expected Revenue":(g.Amount*g["Effective Probability"]).sum(),"Pipeline Amount":g.Amount.sum()}),include_groups=False).reset_index(drop=True)
        c1,c2=st.columns(2); c1.subheader("Funding agency"); c1.bar_chart(agency.set_index("Funding Agency")); c1.dataframe(agency,width="stretch",hide_index=True); c2.subheader("Capture manager"); c2.dataframe(manager.sort_values("Expected Revenue",ascending=False).style.format({"Expected Revenue":"${:,.0f}","Pipeline Amount":"${:,.0f}"}),width="stretch",hide_index=True)
    with tabs[4]: st.dataframe(r["sensitivity"].style.format({"Amount":"${:,.0f}","Effective Probability":"{:.0%}","Revenue Swing":"${:,.0f}","Expected Revenue":"${:,.0f}"}),width="stretch",hide_index=True)
    with tabs[5]:
        st.write(f"Exact modal bid combination occurred in **{r['mode_frequency']:.1%}** of runs."); st.write("**Booked:** "+", ".join(d.loc[d.Status.eq("Booked"),"Program"])); st.write("**Bid wins:** "+(", ".join(r["modal_wins"]) if r["modal_wins"] else "None")); st.dataframe(d[["Program","Status","Opportunity Stage","Effective Probability","Funding Agency","Amount"]].style.format({"Effective Probability":"{:.0%}","Amount":"${:,.0f}"}),width="stretch",hide_index=True)
    with tabs[6]:
        csv=monthly.copy(); csv.Month=csv.Month.dt.strftime("%Y-%m"); settings={"runs":int(st.session_state.runs),"seed":int(st.session_state.seed),"fy_start":int(st.session_state.fy_start)}
        snapshot=d[COLUMNS].copy(); snapshot["Start Date"]=snapshot["Start Date"].dt.strftime("%Y-%m-%d"); snapshot["End Date"]=snapshot["End Date"].dt.strftime("%Y-%m-%d")
        archive={"forecast_name":st.session_state.forecast_name,"description":st.session_state.description,"settings":settings,"programs":json.loads(snapshot.to_json(orient="records"))}
        st.download_button("Download monthly forecast CSV",csv.to_csv(index=False).encode(),"monthly_revenue_forecast.csv","text/csv",width="stretch")
        st.download_button("Download portable scenario JSON",json.dumps(archive,indent=2).encode(),"forecast_scenario.json","application/json",width="stretch")
        st.download_button("Download executive PDF",pdf_report(st.session_state.forecast_name,st.session_state.description,d,r,st.session_state.fy_start),"revenue_forecast_executive_report.pdf","application/pdf",width="stretch")

with st.expander("V2 assumptions"):
    st.markdown("""- Booked programs occur in every run. Each bid is sampled independently.\n- Checked stage probabilities override manually entered P(win).\n- Revenue is spread evenly across inclusive calendar months.\n- SQLite saves are local to the running app instance. On ephemeral hosting, use the portable JSON/CSV downloads for durable retention.\n- P10/P50/P90 reflect award uncertainty only. Delay, funding reduction, and correlated-award modeling are reserved for V3.""")
