"""
Streamlit Web Dashboard for FreightTiger Smart Shipping Cost Assistant.
Features:
- Live KPI Metrics (Total Shipments, Evaluated Route-Weeks, Anomalies, Justified vs Unexplained)
- Interactive Cost-per-Tonne-Km (CPTK) Historical Trend Visualizer
- Anomaly Explorer with Corridor & Verdict Filtering
- Plain-English Interactive Q&A Assistant (Stretch Goal)
- One-Click Deliverables Export
"""

import sys
from pathlib import Path
import pandas as pd
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
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF5500;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #FF5500;
    }
    .badge-yes {
        background-color: #FFEBE6;
        color: #DE350B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-justified {
        background-color: #E3FCEF;
        color: #006644;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
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
    
    return output_df, all_weeks_df, raw_df, notes_df


output_df, all_weeks_df, raw_df, notes_df = load_data()

# Sidebar
st.sidebar.image("https://www.freighttiger.com/wp-content/uploads/2021/04/Freight-Tiger-Logo.svg", width=220)
st.sidebar.markdown("### 🚛 System Configuration")
st.sidebar.info(
    f"**Engine Mode:** Deterministic RAG + Guardrails\n\n"
    f"**Total Shipments:** {len(raw_df):,}\n\n"
    f"**Corridors Tracked:** {all_weeks_df['route'].nunique()}\n\n"
    f"**Context Notes:** {len(notes_df)}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download Deliverables")
if settings.OUTPUT_CSV_PATH.exists():
    with open(settings.OUTPUT_CSV_PATH, "r", encoding="utf-8") as f:
        st.sidebar.download_button(
            label="📄 Download output.csv",
            data=f.read(),
            file_name="output.csv",
            mime="text/csv",
            use_container_width=True
        )

if settings.REPRODUCIBILITY_REPORT_PATH.exists():
    with open(settings.REPRODUCIBILITY_REPORT_PATH, "r", encoding="utf-8") as f:
        st.sidebar.download_button(
            label="📋 Download Reproducibility Report",
            data=f.read(),
            file_name="reproducibility_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# Header
st.markdown('<p class="main-header">FreightTiger Smart Shipping Cost Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Anomaly Detection, RAG Context Grounding & Plain-English Cost Auditing</p>', unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Shipments", f"{len(raw_df):,}")
with col2:
    st.metric("Route-Weeks Tracked", f"{len(all_weeks_df):,}")
with col3:
    st.metric("Cost Anomalies", f"{len(output_df)}")
with col4:
    justified_count = (output_df["flagged"] == "No (justified)").sum()
    st.metric("Justified Rises", f"{justified_count}", delta="Grounded Notes")
with col5:
    unexplained_count = (output_df["flagged"] == "Yes").sum()
    st.metric("Unexplained Flags", f"{unexplained_count}", delta="Review Required", delta_color="inverse")

st.markdown("---")

# Navigation Tabs
tab_anomalies, tab_trends, tab_qa, tab_notes = st.tabs([
    "🚨 Anomaly Explorer",
    "📈 Route CPTK Trends",
    "💬 Interactive Q&A Assistant (Stretch Goal)",
    "📝 Context Notes & Guardrails"
])

# TAB 1: Anomaly Explorer
with tab_anomalies:
    st.subheader("Audited Cost Spike Events")
    st.markdown("All candidate route-weeks where shipping costs rose out of the ordinary compared to history or peers.")
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        route_filter = st.multiselect("Filter by Route:", options=output_df["route"].unique(), default=output_df["route"].unique())
    with col_f2:
        verdict_filter = st.multiselect("Filter by Verdict:", options=output_df["flagged"].unique(), default=output_df["flagged"].unique())
        
    filtered_df = output_df[
        (output_df["route"].isin(route_filter)) &
        (output_df["flagged"].isin(verdict_filter))
    ]
    
    st.dataframe(
        filtered_df,
        column_config={
            "route": st.column_config.TextColumn("Route Corridor", width="medium"),
            "week_of": st.column_config.TextColumn("Week (Monday)", width="small"),
            "cost_per_tonne_km": st.column_config.NumberColumn("CPTK (INR)", format="₹%.2f"),
            "vs_own_history": st.column_config.TextColumn("vs. Own 8w History", width="medium"),
            "vs_similar_routes": st.column_config.TextColumn("vs. Similar Routes", width="medium"),
            "flagged": st.column_config.TextColumn("Flagged Status", width="small"),
            "matched_note_id": st.column_config.TextColumn("Note ID", width="small"),
            "reason": st.column_config.TextColumn("Plain-English Explanation", width="large"),
        },
        hide_index=True,
        use_container_width=True
    )

# TAB 2: Route CPTK Trends
with tab_trends:
    st.subheader("Route-by-Route Cost per Tonne-Km (CPTK) History")
    selected_route = st.selectbox("Select Route to Inspect:", options=all_weeks_df["route"].unique(), index=0)
    
    route_data = all_weeks_df[all_weeks_df["route"] == selected_route].copy()
    route_data["week_of_dt"] = pd.to_datetime(route_data["week_of"])
    route_data = route_data.sort_values("week_of_dt").reset_index(drop=True)
    
    # Plotly or Streamlit Line Chart
    chart_data = route_data.set_index("week_of_dt")[["cost_per_tonne_km"]]
    chart_data.columns = [f"{selected_route} CPTK (INR)"]
    st.line_chart(chart_data)
    
    # Show anomaly weeks on this route
    route_anomalies = output_df[output_df["route"] == selected_route]
    if not route_anomalies.empty:
        st.markdown(f"**Flagged Events on {selected_route}:**")
        st.table(route_anomalies[["week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]])
    else:
        st.success(f"No cost rise anomalies flagged on {selected_route}. Route operated consistently within baselines.")

# TAB 3: Interactive Q&A Assistant
with tab_qa:
    st.subheader("Interactive Freight Cost Assistant (Natural Language Q&A)")
    st.markdown("Ask any question in plain English regarding shipping cost movements, reasons, and baselines.")
    
    sample_questions = [
        "Why did Chennai-Bangalore get pricier in March 2025?",
        "Why did Ahmedabad-Mumbai spike in January 2025?",
        "What happened on Delhi-Jaipur in November 2024?",
        "What routes had unexplained cost increases in 2025?"
    ]
    
    selected_sample = st.selectbox("Or choose a sample question:", ["(Type your own below...)"] + sample_questions)
    
    user_query = st.text_input(
        "Enter your question:",
        value=selected_sample if selected_sample != "(Type your own below...)" else ""
    )
    
    if st.button("Ask Assistant", type="primary"):
        if user_query.strip():
            with st.spinner("Analyzing shipping data and grounded context notes..."):
                agent = ShippingCostQAAgent()
                answer = agent.answer(user_query)
                st.markdown("### Assistant Response:")
                st.markdown(answer)
        else:
            st.warning("Please enter a question.")

# TAB 4: Context Notes & Guardrails
with tab_notes:
    st.subheader("Knowledge Base: Context Notes")
    st.markdown("The 10 real-world context notes used by the RAG system, along with their guardrail audit classifications:")
    
    st.dataframe(
        notes_df,
        column_config={
            "note_id": st.column_config.TextColumn("Note ID", width="small"),
            "date": st.column_config.TextColumn("Date", width="small"),
            "applies_to": st.column_config.TextColumn("Applies To", width="medium"),
            "note": st.column_config.TextColumn("Content", width="large")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("""
    ### Guardrail Hallucination Defense Strategy
    - **N001 (Chennai-Bangalore Flooding):** Valid operational justification for Feb 24 - Mar 8 disruption window.
    - **N002 (Ahmedabad-Mumbai Surcharge):** Valid operational justification for regional festival week of Jan 20, 2025.
    - **N003 (Nationwide Diesel Hike):** Valid operational justification for 5-7% rate rise from May 5, 2025.
    - **N004 (Toll Plaza):** **REJECTED AS JUSTIFICATION.** Explicitly states affected routes are *not* part of this dataset.
    - **N005 (Mumbai-Delhi Maintenance):** **REJECTED AS JUSTIFICATION.** Explicitly states costs were *not* significantly affected.
    - **N006 (Quarterly Report):** **REJECTED AS JUSTIFICATION.** States stable demand with *no major disruptions*.
    - **N007 (Delhi-Jaipur Resurfacing):** **REJECTED AS JUSTIFICATION.** Improved road conditions do not justify cost increase.
    - **N008 (Kolkata-Bhubaneswar Normal):** **REJECTED AS JUSTIFICATION.** Reports normal movement with no disruptions.
    - **N009 (Flood Repair Completed):** **REJECTED AS JUSTIFICATION.** Confirms return to normal conditions.
    - **N010 (Fleet Tracking Mandate):** **REJECTED AS JUSTIFICATION.** Explicitly states compliance costs were *absorbed without rate change*.
    """)
