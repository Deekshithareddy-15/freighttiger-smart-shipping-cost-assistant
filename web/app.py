"""
Streamlit Web Dashboard for FreightTiger Smart Shipping Cost Assistant.
Features:
- Live KPI Metrics (Total Shipments, Evaluated Route-Weeks, Anomalies, Justified vs Unexplained)
- Interactive Cost-per-Tonne-Km (CPTK) Historical Trend Visualizer with Altair
- Anomaly Explorer with Corridor & Verdict Filtering and Shortened Headers
- Plain-English Interactive Q&A Assistant (Stretch Goal)
- Context Notes & Zero-Hallucination Guardrail Decision Matrix
- One-Click Deliverables Export
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.assistant import ShippingCostAssistant
from src.interactive_qa import ShippingCostQAAgent

# Page Configuration
st.set_page_config(
    page_title="FreightTiger AI | Smart Shipping Cost Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Typography & Font Sizing */
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FF5500;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.98rem;
        color: #94A3B8;
        margin-bottom: 22px;
        font-weight: 400;
    }
    
    /* KPI Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        border-top: 3px solid #FF5500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Quick Action Pill */
    .quick-pill {
        display: inline-block;
        background: rgba(255, 85, 0, 0.12);
        color: #FF6611;
        border: 1px solid rgba(255, 85, 0, 0.25);
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Response Card */
    .qa-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 4px solid #FF5500;
        margin-top: 12px;
    }
    
    /* Guardrail Strategy Boxes */
    .guardrail-card-valid {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .guardrail-card-trap {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Loads or generates assistant outputs."""
    if not settings.OUTPUT_CSV_PATH.exists() or not settings.OUTPUT_ALL_WEEKS_CSV_PATH.exists():
        assistant = ShippingCostAssistant()
        assistant.run_pipeline(export_csv=True)
        
    output_df = pd.read_csv(settings.OUTPUT_CSV_PATH)
    all_weeks_df = pd.read_csv(settings.OUTPUT_ALL_WEEKS_CSV_PATH)
    raw_df = pd.read_csv(settings.SHIPMENT_RECORDS_PATH)
    notes_df = pd.read_csv(settings.CONTEXT_NOTES_PATH)
    
    # Ensure numeric types
    output_df["cost_per_tonne_km"] = pd.to_numeric(output_df["cost_per_tonne_km"], errors="coerce")
    all_weeks_df["cost_per_tonne_km"] = pd.to_numeric(all_weeks_df["cost_per_tonne_km"], errors="coerce")
    
    return output_df, all_weeks_df, raw_df, notes_df


output_df, all_weeks_df, raw_df, notes_df = load_data()

# -------------------------------------------------------------
# SIDEBAR: Enterprise Branding & Status
# -------------------------------------------------------------
with st.sidebar:
    # 1. Clean native brand header (eliminates broken external SVG and '0' glitch)
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1);">
        <div style="background: linear-gradient(135deg, #FF5500 0%, #FF7700 100%); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 4px 12px rgba(255,85,0,0.35);">
            🚚
        </div>
        <div>
            <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.1;">
                Freight<span style="color: #FF5500;">Tiger</span>
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">
                Cost Intelligence AI
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ System Status")
    st.info(
        f"**🟢 Status:** Online & Deterministic\n\n"
        f"**🧠 Vector DB:** ChromaDB (Local MiniLM)\n\n"
        f"**📊 Shipments:** {len(raw_df):,} records\n\n"
        f"**🛣️ Corridors:** {all_weeks_df['route'].nunique()} routes\n\n"
        f"**📝 Context Notes:** {len(notes_df)} notes"
    )
    
    st.markdown("---")
    st.markdown("### 📥 Export Deliverables")
    
    if settings.OUTPUT_CSV_PATH.exists():
        with open(settings.OUTPUT_CSV_PATH, "r", encoding="utf-8") as f:
            st.download_button(
                label="📄 Download output.csv",
                data=f.read(),
                file_name="output.csv",
                mime="text/csv",
                width="stretch"
            )
            
    if settings.OUTPUT_ALL_WEEKS_CSV_PATH.exists():
        with open(settings.OUTPUT_ALL_WEEKS_CSV_PATH, "r", encoding="utf-8") as f:
            st.download_button(
                label="📊 Download output_all_weeks.csv",
                data=f.read(),
                file_name="output_all_weeks.csv",
                mime="text/csv",
                width="stretch"
            )

    if settings.REPRODUCIBILITY_REPORT_PATH.exists():
        with open(settings.REPRODUCIBILITY_REPORT_PATH, "r", encoding="utf-8") as f:
            st.download_button(
                label="📋 Download Reproducibility Report",
                data=f.read(),
                file_name="reproducibility_report.txt",
                mime="text/plain",
                width="stretch"
            )

# -------------------------------------------------------------
# MAIN HEADER & KPI CARDS
# -------------------------------------------------------------
st.markdown('<div class="brand-title">FreightTiger Smart Shipping Cost Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-subtitle">'
    'Autonomous Freight Cost Auditing, Trailing 8w & Peer Baselines, RAG Operational Grounding'
    '</div>',
    unsafe_allow_html=True
)

# 5 High-Impact KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Shipments", f"{len(raw_df):,}")
with col2:
    st.metric("Route-Weeks Tracked", f"{len(all_weeks_df):,}")
with col3:
    st.metric("Cost Anomalies", f"{len(output_df)}")
with col4:
    justified_count = int((output_df["flagged"] == "No (justified)").sum())
    st.metric("Justified Rises", f"{justified_count}", delta="Grounded in Notes")
with col5:
    unexplained_count = int((output_df["flagged"] == "Yes").sum())
    st.metric("Unexplained Flags", f"{unexplained_count}", delta="Action Required", delta_color="inverse")

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# NAVIGATION TABS
# -------------------------------------------------------------
tab_anomalies, tab_trends, tab_qa, tab_notes = st.tabs([
    "🚨 Anomaly Explorer",
    "📈 Route CPTK Trends",
    "💬 Interactive Q&A Assistant (Stretch Goal)",
    "📝 Context Notes & Guardrails"
])

# -------------------------------------------------------------
# TAB 1: Anomaly Explorer
# -------------------------------------------------------------
with tab_anomalies:
    st.subheader("Audited Cost Spike Events")
    st.caption("All candidate route-weeks where shipping cost per tonne-km rose out of the ordinary compared to history or peers.")
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        route_filter = st.multiselect(
            "Filter by Corridor:",
            options=sorted(output_df["route"].unique()),
            default=sorted(output_df["route"].unique())
        )
    with col_f2:
        verdict_filter = st.multiselect(
            "Filter by Audit Status:",
            options=sorted(output_df["flagged"].unique()),
            default=sorted(output_df["flagged"].unique())
        )
        
    filtered_df = output_df[
        (output_df["route"].isin(route_filter)) &
        (output_df["flagged"].isin(verdict_filter))
    ].copy()
    
    # Quick Summary Status Pills
    f_total = len(filtered_df)
    f_justified = (filtered_df["flagged"] == "No (justified)").sum()
    f_unexplained = (filtered_df["flagged"] == "Yes").sum()
    
    st.markdown(
        f"<div style='margin: 10px 0 14px 0;'>"
        f"<span class='quick-pill'>Filtered Events: {f_total}</span>"
        f"<span class='quick-pill' style='color:#10B981; background:rgba(16,185,129,0.1); border-color:rgba(16,185,129,0.3);'>🟢 Justified: {f_justified}</span>"
        f"<span class='quick-pill' style='color:#EF4444; background:rgba(239,68,68,0.1); border-color:rgba(239,68,68,0.3);'>🔴 Unexplained: {f_unexplained}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Shortened column headers with clear widths and horizontal scroll
    st.dataframe(
        filtered_df,
        column_config={
            "route": st.column_config.TextColumn("Route", width=160),
            "week_of": st.column_config.TextColumn("Week (Mon)", width=110),
            "cost_per_tonne_km": st.column_config.NumberColumn("CPTK (₹)", format="₹%.2f", width=100),
            "vs_own_history": st.column_config.TextColumn("vs. Own 8w Avg", width=140),
            "vs_similar_routes": st.column_config.TextColumn("vs. Peer Avg", width=140),
            "flagged": st.column_config.TextColumn("Status", width=120),
            "matched_note_id": st.column_config.TextColumn("Note ID", width=90),
            "reason": st.column_config.TextColumn("Explanation & Operational Rationale", width=380),
        },
        hide_index=True,
        width="stretch"
    )

# -------------------------------------------------------------
# TAB 2: Route CPTK Trends (Altair Layered Chart)
# -------------------------------------------------------------
with tab_trends:
    st.subheader("Route-by-Route Cost per Tonne-Km (CPTK) History")
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_route = st.selectbox(
            "Select Route Corridor:",
            options=sorted(all_weeks_df["route"].unique()),
            index=0
        )
        
    route_all = all_weeks_df[all_weeks_df["route"] == selected_route].copy()
    route_all["week_of_dt"] = pd.to_datetime(route_all["week_of"])
    route_all = route_all.sort_values("week_of_dt").reset_index(drop=True)
    
    # Identify verified anomaly records on this route from output_df
    route_anomalies = output_df[output_df["route"] == selected_route].copy()
    
    # Select clean base columns from route_all (derived from all_weeks_df)
    base_cols = ["route", "week_of", "week_of_dt", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes"]
    available_base_cols = [c for c in base_cols if c in route_all.columns]
    route_chart_base = route_all[available_base_cols].copy()
    
    # Merge with verified anomalies strictly on BOTH 'route' and 'week_of'
    anomaly_cols = ["route", "week_of", "flagged", "matched_note_id", "reason"]
    available_anomaly_cols = [c for c in anomaly_cols if c in route_anomalies.columns]
    
    route_chart_df = route_chart_base.merge(
        route_anomalies[available_anomaly_cols],
        on=["route", "week_of"],
        how="left"
    )
    
    # Safe derivation of audit status:
    # A week is a verified 'Normal Baseline' ONLY if cost_per_tonne_km is present and positive,
    # and no anomaly was recorded for that corridor and week.
    def derive_audit_status(row):
        flag = row.get("flagged")
        note_id = row.get("matched_note_id")
        
        if pd.notna(flag) and str(flag).strip() != "":
            flag_str = str(flag).strip()
            if flag_str == "No (justified)":
                note_str = str(note_id).strip() if pd.notna(note_id) and str(note_id).strip() else ""
                return f"Justified (Note {note_str})" if note_str else "Justified Rise"
            elif flag_str == "Yes":
                return "Unexplained Spike"
            return flag_str
            
        cptk_val = row.get("cost_per_tonne_km")
        if pd.notna(cptk_val) and cptk_val > 0:
            return "Normal Baseline"
        return "Unevaluated / No Data"

    route_chart_df["audit_status"] = route_chart_df.apply(derive_audit_status, axis=1)
    route_chart_df["is_anomaly"] = route_chart_df["flagged"].notna() & route_chart_df["flagged"].isin(["Yes", "No (justified)"])
    
    with col_sel2:
        route_type = raw_df[raw_df["origin"] + "-" + raw_df["destination"] == selected_route]["route_type"].iloc[0]
        avg_cptk = route_all["cost_per_tonne_km"].mean()
        st.markdown(
            f"<div style='margin-top: 14px;'>"
            f"<span class='quick-pill'>Length Bucket: {route_type}</span>"
            f"<span class='quick-pill'>Average CPTK: ₹{avg_cptk:.2f}</span>"
            f"<span class='quick-pill'>Flagged Spikes: {len(route_anomalies)}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Build Altair Layered Chart with interactive tooltip
    x_axis = alt.X(
        "week_of_dt:T",
        title="Shipment Week (Monday)",
        axis=alt.Axis(format="%b %Y", labelAngle=-30, tickCount=12, grid=True, gridColor="rgba(255,255,255,0.06)")
    )
    y_axis = alt.Y(
        "cost_per_tonne_km:Q",
        title="Cost per Tonne-Km (₹)",
        scale=alt.Scale(zero=False, padding=15),
        axis=alt.Axis(grid=True, gridColor="rgba(255,255,255,0.06)")
    )
    
    # 1. Main CPTK trend line
    line = alt.Chart(route_chart_df).mark_line(color="#FF5500", strokeWidth=2.2).encode(
        x=x_axis,
        y=y_axis
    )
    
    # 2. Hover points with full tooltip
    points = alt.Chart(route_chart_df).mark_circle(size=40, color="#FF5500", opacity=0.7).encode(
        x=x_axis,
        y=y_axis,
        tooltip=[
            alt.Tooltip("week_of_dt:T", title="Week (Monday)", format="%Y-%m-%d"),
            alt.Tooltip("cost_per_tonne_km:Q", title="CPTK (₹)", format=".2f"),
            alt.Tooltip("vs_own_history:N", title="vs. Own 8w History"),
            alt.Tooltip("vs_similar_routes:N", title="vs. Similar Routes"),
            alt.Tooltip("audit_status:N", title="Audit Status")
        ]
    )
    
    # 3. Prominent anomaly spike markers
    anomaly_points_df = route_chart_df[route_chart_df["is_anomaly"]]
    if not anomaly_points_df.empty:
        alert_markers = alt.Chart(anomaly_points_df).mark_point(
            shape="diamond",
            size=140,
            fill="#EF4444",
            stroke="#FFFFFF",
            strokeWidth=1.5
        ).encode(
            x=x_axis,
            y=y_axis,
            tooltip=[
                alt.Tooltip("week_of_dt:T", title="ALERT WEEK", format="%Y-%m-%d"),
                alt.Tooltip("cost_per_tonne_km:Q", title="Spike CPTK (₹)", format=".2f"),
                alt.Tooltip("vs_own_history:N", title="vs. Own History"),
                alt.Tooltip("audit_status:N", title="Audit Verdict"),
                alt.Tooltip("matched_note_id:N", title="Matched Note")
            ]
        )
        final_chart = (line + points + alert_markers).properties(height=360)
    else:
        final_chart = (line + points).properties(height=360)
        
    st.altair_chart(final_chart, width="stretch")
    
    # Display table of anomalies for this corridor
    if not route_anomalies.empty:
        st.markdown(f"**Identified Cost Anomalies on {selected_route}:**")
        st.dataframe(
            route_anomalies,
            column_config={
                "route": st.column_config.TextColumn("Route", width=150),
                "week_of": st.column_config.TextColumn("Week", width=110),
                "cost_per_tonne_km": st.column_config.NumberColumn("CPTK (₹)", format="₹%.2f", width=100),
                "vs_own_history": st.column_config.TextColumn("vs. Own 8w", width=130),
                "vs_similar_routes": st.column_config.TextColumn("vs. Peers", width=130),
                "flagged": st.column_config.TextColumn("Status", width=110),
                "matched_note_id": st.column_config.TextColumn("Note ID", width=90),
                "reason": st.column_config.TextColumn("Explanation", width=360),
            },
            hide_index=True,
            width="stretch"
        )
    else:
        st.success(f"No cost rise anomalies detected on {selected_route}. Corridors operated consistently within baselines.")

# -------------------------------------------------------------
# TAB 3: Interactive Q&A Assistant (Stretch Goal)
# -------------------------------------------------------------
with tab_qa:
    st.subheader("Interactive Freight Cost Assistant")
    st.caption("Ask natural language questions about shipping cost trends, corridor spikes, and operational reasons.")
    
    sample_queries = [
        "Why did Chennai-Bangalore get pricier in March 2025?",
        "Why did Ahmedabad-Mumbai spike in January 2025?",
        "What happened on Delhi-Jaipur in November 2024?",
        "What routes had unexplained cost increases in 2025?"
    ]
    
    st.markdown("**Sample Questions:**")
    cols_s = st.columns(len(sample_queries))
    selected_sample = None
    for i, sq in enumerate(sample_queries):
        if cols_s[i].button(sq, key=f"sq_{i}", width="stretch"):
            selected_sample = sq
            
    # Form input container
    with st.form("qa_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter your question:",
            value=selected_sample or "",
            placeholder="e.g. Why did Chennai-Bangalore cost rise in February 2025?"
        )
        submit_btn = st.form_submit_button("Ask Assistant", type="primary")
        
    if submit_btn or selected_sample:
        query_to_run = user_query.strip() or (selected_sample or "").strip()
        if query_to_run:
            with st.spinner("Retrieving corridor baselines and grounded context notes..."):
                agent = ShippingCostQAAgent()
                answer = agent.answer(query_to_run)
                
                st.markdown(
                    f"""
                    <div class="qa-card">
                        <div style="font-size: 0.82rem; color: #FF5500; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px;">
                            🤖 Assistant Response
                        </div>
                        <div style="font-size: 0.95rem; line-height: 1.6;">
                            {answer.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter or select a question to ask.")

# -------------------------------------------------------------
# TAB 4: Context Notes & Guardrails
# -------------------------------------------------------------
with tab_notes:
    st.subheader("Operational Context Notes & Guardrails")
    st.caption("The 10 real-world context notes used by the RAG system and their strict guardrail audit classifications.")
    
    st.dataframe(
        notes_df,
        column_config={
            "note_id": st.column_config.TextColumn("Note ID", width=90),
            "date": st.column_config.TextColumn("Date", width=110),
            "applies_to": st.column_config.TextColumn("Corridor Scope", width=160),
            "note": st.column_config.TextColumn("Operational Note Content", width=480),
        },
        hide_index=True,
        width="stretch"
    )
    
    st.markdown("### 🛡️ Guardrail Decision Matrix (Zero-Hallucination Policy)")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        <div class="guardrail-card-valid">
            <h4 style="color: #10B981; margin-top: 0;">🟢 Valid Operational Justifications</h4>
            <ul>
                <li><b>N001 (Chennai-Bangalore):</b> Flooding from Feb 24 - Mar 8, 2025 forced detours and higher trip costs. Valid for disruption window.</li>
                <li><b>N002 (Ahmedabad-Mumbai):</b> Regional festival week surcharge on Jan 20, 2025 due to peak demand and limited truck availability.</li>
                <li><b>N003 (All Routes):</b> Nationwide diesel price increase on May 5, 2025 pushed transportation costs up by 5-7%.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_g2:
        st.markdown("""
        <div class="guardrail-card-trap">
            <h4 style="color: #EF4444; margin-top: 0;">🔴 Strictly Rejected Trap Notes</h4>
            <ul>
                <li><b>N004 (Toll Plaza):</b> <i>Rejected.</i> Affected routes are explicitly not part of this dataset.</li>
                <li><b>N005 (Mumbai-Delhi):</b> <i>Rejected.</i> Note explicitly confirms costs were not significantly affected.</li>
                <li><b>N006 (All Routes):</b> <i>Rejected.</i> Mentions stable demand with no major disruptions (Sample Row 4).</li>
                <li><b>N007 (Delhi-Jaipur):</b> <i>Rejected.</i> Road resurfacing does not justify price increases.</li>
                <li><b>N008 (Kolkata-Bhubaneswar):</b> <i>Rejected.</i> Reports normal movement with no disruptions.</li>
                <li><b>N009 (Chennai-Bangalore):</b> <i>Rejected.</i> Confirms return to normal conditions.</li>
                <li><b>N010 (All Routes):</b> <i>Rejected.</i> Fleet tracking compliance costs were absorbed without rate change.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
