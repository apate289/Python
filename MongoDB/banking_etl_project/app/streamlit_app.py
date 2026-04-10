"""
Banking Client Intelligence — Streamlit Dashboard
Connects to MongoDB to visualize active/historical records with SCD Type 2 versioning.

Run:  streamlit run app/streamlit_app.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.queries import (
    get_db, get_kpis, get_clients, get_client_by_id,
    get_client_version_history, get_audit_logs,
    get_account_type_distribution, get_credit_score_buckets,
    get_state_distribution, get_branch_summary,
    get_version_history_stats, get_filter_options,
    # hash comparison
    get_diffs_for_client, get_client_hash_manifest,
    get_most_changed_fields, get_most_changed_clients,
    get_section_change_frequency, get_diff_timeline,
)
from etl.hash_engine import build_hash_manifest, SECTIONS, TRACKED_FIELDS

# ── page config ──────────────────────────────
st.set_page_config(
    page_title="Banking Client Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0f1e; }
section[data-testid="stSidebar"] { background: #0d1526 !important; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7fa3;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #e8f0fe;
    font-family: 'Space Mono', monospace;
}
.metric-delta {
    font-size: 12px;
    color: #34d399;
    margin-top: 4px;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a90d9;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}
.badge-active   { background:#064e3b; color:#6ee7b7; padding:2px 10px; border-radius:20px; font-size:11px; }
.badge-inactive { background:#451a03; color:#fbbf24; padding:2px 10px; border-radius:20px; font-size:11px; }
.badge-kyc      { background:#1e3a5f; color:#93c5fd; padding:2px 10px; border-radius:20px; font-size:11px; }
.version-pill   { background:#312e81; color:#a5b4fc; padding:2px 8px; border-radius:12px; font-size:11px; font-family:'Space Mono',monospace; }
.client-card    { background:#111827; border:1px solid #1e3a5f; border-radius:10px; padding:16px; margin-bottom:10px; }

[data-testid="stMetric"] label { color: #6b7fa3 !important; font-size: 12px !important; }
[data-testid="stMetric"] [data-testid="metric-container"] { background: #111827; border-radius: 10px; padding: 16px; border: 1px solid #1e3a5f; }

.stButton>button {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
}
div.stSelectbox label, div.stSlider label, div.stTextInput label,
div.stMultiSelect label, div.stCheckbox label { color: #94a3b8 !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    margin=dict(l=10, r=10, t=30, b=10),
    colorway=["#3b82f6","#6366f1","#8b5cf6","#a78bfa","#34d399","#f59e0b"],
)

# ── db connection ─────────────────────────────
@st.cache_resource
def init_db():
    return get_db()

db = init_db()

# ════════════════════════════════════════════════
#  SIDEBAR — filters
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Banking Intel")
    st.markdown("---")

    options = get_filter_options(db)

    view_mode = st.radio("Record View", ["Active Records", "Historical Records", "All Versions"], index=0)
    is_active_filter = True if view_mode == "Active Records" else (False if view_mode == "Historical Records" else None)

    st.markdown("### Filters")
    search_q   = st.text_input("🔍 Search (ID / Name / Email)", placeholder="CLT00001 or Michael…")
    state_sel  = st.selectbox("State", ["All"] + options["states"])
    branch_sel = st.selectbox("Branch", ["All"] + options["branches"])
    acct_sel   = st.selectbox("Account Type", ["All"] + options["account_types"])
    acct_status_sel = st.selectbox("Account Status", ["All", "Active", "Dormant", "Closed", "Frozen"])

    kyc_sel = st.selectbox("KYC Verified", ["All", "Yes", "No"])
    pep_sel = st.selectbox("PEP Flag", ["All", "Yes", "No"])

    credit_range = st.slider("Credit Score Range", 300, 850, (300, 850))

    page_size = st.select_slider("Records per page", options=[10, 25, 50, 100], value=25)

    st.markdown("---")
    nav = st.radio("Navigation", ["📊 Dashboard", "👥 Client Explorer", "🕓 Version History", "🔍 Hash Comparison", "📋 ETL Audit Log"])

# ── filter translation ─────────────────────────
kyc_bool  = True if kyc_sel == "Yes" else (False if kyc_sel == "No" else None)
pep_bool  = True if pep_sel == "Yes" else (False if pep_sel == "No" else None)
state_v   = None if state_sel  == "All" else state_sel
branch_v  = None if branch_sel == "All" else branch_sel
acct_v    = None if acct_sel   == "All" else acct_sel
astatus_v = None if acct_status_sel == "All" else acct_status_sel
search_v  = search_q.strip() or None

# ════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════
if nav == "📊 Dashboard":
    st.markdown("<h1 style='color:#e8f0fe;font-family:Space Mono,monospace;font-size:28px;'>Banking Client Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7fa3;margin-top:-12px;'>Real-time MongoDB Analytics · SCD Type 2 Versioning</p>", unsafe_allow_html=True)

    # KPI row
    kpis = get_kpis(db)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Total Active Clients",  f"{kpis['total_clients']:,}")
    with k2: st.metric("KYC Verified",          f"{kpis['kyc_verified']:,}")
    with k3: st.metric("PEP Flagged",           f"{kpis['pep_flagged']:,}")
    with k4: st.metric("Avg Credit Score",      f"{kpis['avg_credit']:,}")
    with k5: st.metric("Total Balance (Active)", f"${kpis['total_balance']:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Account type distribution
    with col1:
        st.markdown("<div class='section-header'>Account Type Distribution</div>", unsafe_allow_html=True)
        acct_data = get_account_type_distribution(db)
        if acct_data:
            df_acct = pd.DataFrame(acct_data).rename(columns={"_id": "Account Type"})
            fig = px.bar(df_acct, x="Account Type", y="count", color="total_balance",
                         color_continuous_scale=["#1e3a5f","#3b82f6","#93c5fd"],
                         labels={"count": "# Accounts", "total_balance": "Total Balance"})
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Credit score buckets
    with col2:
        st.markdown("<div class='section-header'>Credit Score Distribution</div>", unsafe_allow_html=True)
        cs_data = get_credit_score_buckets(db)
        if cs_data:
            df_cs = pd.DataFrame(cs_data)
            fig = px.pie(df_cs, names="label", values="count",
                         color_discrete_sequence=["#ef4444","#f59e0b","#3b82f6","#6366f1","#34d399"])
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              hole=0.45, textfont_size=11)
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # State heatmap
    with col3:
        st.markdown("<div class='section-header'>Clients by State</div>", unsafe_allow_html=True)
        state_data = get_state_distribution(db)
        if state_data:
            df_state = pd.DataFrame(state_data).rename(columns={"_id": "state"})
            fig = px.choropleth(df_state, locations="state", locationmode="USA-states",
                                color="count", scope="usa",
                                color_continuous_scale=["#0a0f1e","#1e3a5f","#3b82f6","#93c5fd"])
            fig.update_layout(**PLOTLY_LAYOUT, geo=dict(bgcolor="rgba(0,0,0,0)",
                              lakecolor="rgba(0,0,0,0)", landcolor="#111827",
                              subunitcolor="#1e3a5f"))
            st.plotly_chart(fig, use_container_width=True)

    # Branch performance
    with col4:
        st.markdown("<div class='section-header'>Top Branches by Client Count</div>", unsafe_allow_html=True)
        branch_data = get_branch_summary(db)
        if branch_data:
            df_br = pd.DataFrame(branch_data).rename(columns={"_id":"Branch","client_count":"Clients","avg_credit":"Avg Credit"})
            df_br["Avg Credit"] = df_br["Avg Credit"].round(0)
            fig = go.Figure(go.Bar(
                x=df_br["Branch"], y=df_br["Clients"],
                marker=dict(color=df_br["Avg Credit"],
                            colorscale=["#ef4444","#f59e0b","#3b82f6","#34d399"],
                            showscale=True,
                            colorbar=dict(title="Avg Credit", thickness=10, tickfont=dict(size=9))),
                text=df_br["Clients"], textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # Version history
    st.markdown("<div class='section-header'>SCD Type 2 — Version Distribution</div>", unsafe_allow_html=True)
    ver_data = get_version_history_stats(db)
    if ver_data:
        df_ver = pd.DataFrame(ver_data).rename(columns={"_id": "Version"})
        fig = px.bar(df_ver, x="Version", y="count",
                     color_discrete_sequence=["#6366f1"],
                     labels={"count": "Record Count", "Version": "Version Number"},
                     title="Records by Version (all time)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE: CLIENT EXPLORER
# ════════════════════════════════════════════════
elif nav == "👥 Client Explorer":
    st.markdown("<h2 style='color:#e8f0fe;font-family:Space Mono,monospace;'>Client Explorer</h2>", unsafe_allow_html=True)

    if "page" not in st.session_state: st.session_state.page = 0

    docs, total = get_clients(
        db,
        is_active=is_active_filter if is_active_filter is not None else True,
        state=state_v, kyc=kyc_bool, pep=pep_bool,
        min_credit=credit_range[0], max_credit=credit_range[1],
        account_type=acct_v, account_status=astatus_v,
        search=search_v, branch_id=branch_v,
        skip=st.session_state.page * page_size,
        limit=page_size,
    )

    st.caption(f"**{total:,}** records matched · Page {st.session_state.page+1} of {max(1, -(-total//page_size))}")

    if docs:
        rows = []
        for d in docs:
            pi  = d.get("personal_information", {})
            bp  = d.get("banking_profile", {})
            acc = d.get("accounts", [])
            rows.append({
                "Client ID":     d.get("client_id"),
                "Name":          f"{pi.get('first_name','')} {pi.get('last_name','')}",
                "State":         d.get("address", {}).get("state"),
                "Branch":        bp.get("branch_id"),
                "Credit Score":  bp.get("credit_score"),
                "KYC":           "✅" if bp.get("kyc_verified") else "❌",
                "PEP":           "🚨" if bp.get("pep_flag") else "—",
                "# Accounts":    len(acc),
                "Active Balance":f"${sum(a['balance'] for a in acc if a.get('status')=='Active'):,.0f}",
                "Version":       d.get("version"),
                "Active":        "🟢 Active" if d.get("is_active") else "🔴 Archived",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Credit Score": st.column_config.ProgressColumn("Credit Score", min_value=300, max_value=850, format="%d"),
                         "Active Balance": st.column_config.TextColumn("Active Balance"),
                     })

        # pagination
        p1, p2, p3 = st.columns([1,3,1])
        with p1:
            if st.button("◀ Prev", disabled=st.session_state.page == 0):
                st.session_state.page -= 1; st.rerun()
        with p3:
            if st.button("Next ▶", disabled=(st.session_state.page + 1) * page_size >= total):
                st.session_state.page += 1; st.rerun()
    else:
        st.info("No records match the current filters.")

    # ── Client Detail Drill-down ──────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>Client Detail View</div>", unsafe_allow_html=True)
    detail_id = st.text_input("Enter Client ID for full detail", placeholder="CLT00001")
    if detail_id:
        client = get_client_by_id(db, detail_id.strip().upper())
        if client:
            pi, addr, bp = client.get("personal_information",{}), client.get("address",{}), client.get("banking_profile",{})
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Personal Information**")
                st.json(pi)
            with c2:
                st.markdown("**Address**")
                st.json(addr)
            with c3:
                st.markdown("**Banking Profile**")
                st.json(bp)

            st.markdown("**Accounts**")
            acc_df = pd.DataFrame(client.get("accounts", []))
            if not acc_df.empty:
                st.dataframe(acc_df, use_container_width=True, hide_index=True)

            cp = client.get("check_printing", {})
            if cp:
                st.markdown(f"**Check Printing** — Authorized: {'✅' if cp.get('authorized') else '❌'} · Style: `{cp.get('check_style')}` · Last Order: `{cp.get('last_check_order_date')}`")
        else:
            st.warning(f"No active client found for `{detail_id}`")


# ════════════════════════════════════════════════
#  PAGE: VERSION HISTORY
# ════════════════════════════════════════════════
elif nav == "🕓 Version History":
    st.markdown("<h2 style='color:#e8f0fe;font-family:Space Mono,monospace;'>SCD Type 2 — Version History</h2>", unsafe_allow_html=True)
    st.markdown("Each time a client record changes during ETL, the previous version is archived and a new version is created.")

    vh_id = st.text_input("Client ID to explore", placeholder="CLT00001")
    if vh_id:
        history = get_client_version_history(db, vh_id.strip().upper())
        if history:
            st.success(f"Found **{len(history)}** version(s) for `{vh_id}`")
            for rec in history:
                v     = rec.get("version", "?")
                active = rec.get("is_active", False)
                badge = "🟢 ACTIVE" if active else "🔴 ARCHIVED"
                with st.expander(f"Version {v}  ·  {badge}  ·  Start: {rec.get('start_date','?')[:10]}  →  End: {rec.get('end_date','—') or '—'}"):
                    t1, t2 = st.columns(2)
                    with t1:
                        st.markdown("**Personal + Address**")
                        st.json({**rec.get("personal_information",{}), **rec.get("address",{})})
                    with t2:
                        st.markdown("**Banking Profile**")
                        st.json(rec.get("banking_profile",{}))
                    st.markdown("**Accounts**")
                    acc_df = pd.DataFrame(rec.get("accounts",[]))
                    if not acc_df.empty: st.dataframe(acc_df, use_container_width=True, hide_index=True)
                    st.caption(f"ETL Batch: `{rec.get('etl_batch_id','—')}`  ·  Updated: `{rec.get('updated_at','—')}`")
        else:
            st.warning(f"No history found for `{vh_id}`")

    # Stat chart
    st.markdown("<div class='section-header'>Records with Multiple Versions</div>", unsafe_allow_html=True)
    ver_data = get_version_history_stats(db)
    if ver_data:
        df_v = pd.DataFrame(ver_data).rename(columns={"_id":"Version","count":"Record Count"})
        fig = px.bar(df_v, x="Version", y="Record Count",
                     color_discrete_sequence=["#8b5cf6"],
                     text="Record Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE: HASH COMPARISON
# ════════════════════════════════════════════════
elif nav == "🔍 Hash Comparison":
    st.markdown("<h2 style='color:#e8f0fe;font-family:Space Mono,monospace;'>Field-Level Hash Comparison</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6b7fa3;margin-top:-10px;'>Every field, section, and account carries its own SHA-256 fingerprint. "
        "Changes are detected and stored at the document → section → field → account-field hierarchy.</p>",
        unsafe_allow_html=True
    )

    hash_tab1, hash_tab2, hash_tab3 = st.tabs(["🔬 Client Diff Explorer", "📊 Change Analytics", "🗂️ Hash Manifest Viewer"])

    # ── Tab 1: Client Diff Explorer ───────────────
    with hash_tab1:
        st.markdown("<div class='section-header'>Select Client to Inspect</div>", unsafe_allow_html=True)
        diff_client_id = st.text_input("Client ID", placeholder="CLT00001", key="diff_cid").strip().upper()

        if diff_client_id:
            diffs = get_diffs_for_client(db, diff_client_id)
            history = get_client_version_history(db, diff_client_id)

            if not history:
                st.warning(f"No records found for `{diff_client_id}`")
            elif not diffs:
                st.info(f"Client `{diff_client_id}` has only one version — no diffs yet.")
                rec = history[0]
                pi = rec.get("personal_information", {})
                st.success(f"**{pi.get('first_name','')} {pi.get('last_name','')}** — v1 (initial load, no changes recorded)")
            else:
                pi = history[0].get("personal_information", {})
                st.success(f"**{pi.get('first_name','')} {pi.get('last_name','')}** — {len(diffs)} version transition(s) found")

                for diff in diffs:
                    ov, nv = diff["old_version"], diff["new_version"]
                    n_fld  = diff.get("total_field_changes", 0)
                    n_acc  = diff.get("total_account_changes", 0)
                    secs   = diff.get("changed_sections", [])

                    with st.expander(
                        f"v{ov} → v{nv}  ·  {len(secs)} section(s)  ·  {n_fld} field(s)  ·  {n_acc} account(s)  ·  {diff.get('run_date','')[:10]}",
                        expanded=(ov == diffs[-1]["old_version"])
                    ):
                        # Document hashes
                        c_oh, c_nh = st.columns(2)
                        with c_oh:
                            st.markdown("**Old document hash (v{})** ".format(ov))
                            st.code(diff.get("old_document_hash", "—"), language=None)
                        with c_nh:
                            st.markdown("**New document hash (v{})** ".format(nv))
                            st.code(diff.get("new_document_hash", "—"), language=None)

                        # Changed sections as badges
                        if secs:
                            st.markdown("**Changed sections:** " + "  ".join(
                                f"`{s}`" for s in secs
                            ))

                        # Field-level changes table
                        field_changes = diff.get("field_changes", [])
                        if field_changes:
                            st.markdown("**Field-level changes**")
                            rows = []
                            for fc in field_changes:
                                path  = fc["path"]
                                section = path.split(".")[0]
                                field   = ".".join(path.split(".")[1:])
                                rows.append({
                                    "Section":   section,
                                    "Field":     field,
                                    "Old Value": fc.get("old_value", "—"),
                                    "New Value": fc.get("new_value", "—"),
                                    "Old Hash":  (fc.get("old_hash") or "")[:20] + "…",
                                    "New Hash":  (fc.get("new_hash") or "")[:20] + "…",
                                })
                            df_fld = pd.DataFrame(rows)
                            st.dataframe(
                                df_fld,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Old Value": st.column_config.TextColumn("Old Value", width="medium"),
                                    "New Value": st.column_config.TextColumn("New Value", width="medium"),
                                    "Old Hash":  st.column_config.TextColumn("Old Hash (SHA-256[:20])", width="medium"),
                                    "New Hash":  st.column_config.TextColumn("New Hash (SHA-256[:20])", width="medium"),
                                }
                            )

                        # Account-level changes
                        acct_changes = diff.get("account_changes", [])
                        if acct_changes:
                            st.markdown("**Account-level changes**")
                            for ac in acct_changes:
                                ct = ac.get("change_type", "modified").upper()
                                badge = {"MODIFIED": "🔄", "ADDED": "✅", "REMOVED": "❌"}.get(ct, "🔄")
                                st.markdown(f"{badge} **Account `{ac['account_number']}`** — {ct}")
                                afc_list = ac.get("field_changes", [])
                                if afc_list:
                                    arows = []
                                    for afc in afc_list:
                                        fn = afc["path"].split(".")[-1]
                                        arows.append({
                                            "Account Field": fn,
                                            "Old Value":     afc.get("old_value", "—"),
                                            "New Value":     afc.get("new_value", "—"),
                                            "Old Hash":      (afc.get("old_hash") or "")[:20] + "…",
                                            "New Hash":      (afc.get("new_hash") or "")[:20] + "…",
                                        })
                                    st.dataframe(pd.DataFrame(arows), use_container_width=True, hide_index=True)

    # ── Tab 2: Change Analytics ───────────────────
    with hash_tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-header'>Most Changed Fields (all clients)</div>", unsafe_allow_html=True)
            mcf = get_most_changed_fields(db)
            if mcf:
                df_mcf = pd.DataFrame(mcf).rename(columns={"_id": "Field Path", "count": "Change Count"})
                fig = px.bar(
                    df_mcf, x="Change Count", y="Field Path", orientation="h",
                    color="Change Count",
                    color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"],
                )
                fig.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No diff data yet. Run the ETL with `--simulate-changes`.")

        with col_b:
            st.markdown("<div class='section-header'>Most Changed Sections</div>", unsafe_allow_html=True)
            scf = get_section_change_frequency(db)
            if scf:
                df_scf = pd.DataFrame(scf).rename(columns={"_id": "Section", "count": "Times Changed"})
                fig2 = px.pie(
                    df_scf, names="Section", values="Times Changed",
                    color_discrete_sequence=["#3b82f6","#6366f1","#8b5cf6","#34d399","#f59e0b"],
                    hole=0.5,
                )
                fig2.update_traces(textposition="inside", textinfo="percent+label")
                fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No diff data yet.")

        st.markdown("<div class='section-header'>Most Changed Clients</div>", unsafe_allow_html=True)
        mcc = get_most_changed_clients(db)
        if mcc:
            df_mcc = pd.DataFrame(mcc).rename(columns={
                "_id": "Client ID",
                "total_versions": "Versions",
                "total_field_changes": "Field Changes",
                "total_acct_changes": "Account Changes",
            })
            df_mcc = df_mcc[["Client ID","Versions","Field Changes","Account Changes"]]
            st.dataframe(
                df_mcc,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field Changes":   st.column_config.ProgressColumn("Field Changes",   min_value=0, max_value=int(df_mcc["Field Changes"].max() or 1)),
                    "Account Changes": st.column_config.ProgressColumn("Account Changes", min_value=0, max_value=max(1, int(df_mcc["Account Changes"].max() or 1))),
                }
            )

        st.markdown("<div class='section-header'>Recent Diff Timeline</div>", unsafe_allow_html=True)
        timeline = get_diff_timeline(db)
        if timeline:
            rows = []
            for d in timeline:
                rows.append({
                    "Date":     d.get("run_date","")[:10],
                    "Client":   d.get("client_id"),
                    "v":        f"v{d.get('old_version')}→v{d.get('new_version')}",
                    "Sections": ", ".join(d.get("changed_sections", [])) or "—",
                    "Fields Δ": d.get("total_field_changes", 0),
                    "Accts Δ":  d.get("total_account_changes", 0),
                    "Old Doc Hash": (d.get("old_document_hash") or "")[:16] + "…",
                    "New Doc Hash": (d.get("new_document_hash") or "")[:16] + "…",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No diff timeline data yet.")

    # ── Tab 3: Hash Manifest Viewer ───────────────
    with hash_tab3:
        st.markdown(
            "<p style='color:#6b7fa3;font-size:13px;'>Inspect the full SHA-256 hash manifest stored inside any client version. "
            "Every field, section, account, and the whole document has its own hash.</p>",
            unsafe_allow_html=True
        )

        hm_col1, hm_col2 = st.columns([2, 1])
        with hm_col1:
            hm_client = st.text_input("Client ID", placeholder="CLT00001", key="hm_cid").strip().upper()
        with hm_col2:
            hm_version = st.number_input("Version (0 = active)", min_value=0, value=0, step=1, key="hm_ver")

        if hm_client:
            ver_arg = int(hm_version) if hm_version > 0 else None
            manifest_doc = get_client_hash_manifest(db, hm_client, version=ver_arg)

            if not manifest_doc:
                st.warning(f"No record found for `{hm_client}` version={hm_version or 'active'}")
            else:
                hashes = manifest_doc.get("_hashes", {})
                ver_label = manifest_doc.get("version", "?")

                if not hashes:
                    st.info("This record has no `_hashes` manifest — it was loaded before the hash engine was added. Re-run the ETL to backfill.")
                else:
                    st.success(f"Showing hash manifest for `{hm_client}` — version {ver_label}")

                    # Document hash
                    st.markdown("**Document hash (whole record)**")
                    st.code(hashes.get("document", "—"), language=None)

                    # Section hashes
                    st.markdown("**Section hashes**")
                    sec_rows = [
                        {"Section": sec, "SHA-256": hashes.get("sections", {}).get(sec, "—")}
                        for sec in SECTIONS
                    ]
                    st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

                    # Field hashes
                    st.markdown("**Leaf field hashes**")
                    fld_rows = []
                    for dotted in TRACKED_FIELDS:
                        section = dotted.split(".")[0]
                        fname   = ".".join(dotted.split(".")[1:])
                        h       = hashes.get("fields", {}).get(dotted, "—")
                        fld_rows.append({"Section": section, "Field": fname, "SHA-256": h})
                    st.dataframe(pd.DataFrame(fld_rows), use_container_width=True, hide_index=True)

                    # Account hashes
                    acct_hashes = hashes.get("accounts", [])
                    if acct_hashes:
                        st.markdown("**Account hashes**")
                        for ah in acct_hashes:
                            with st.expander(f"Account `{ah['account_number']}` — hash: `{ah['hash'][:24]}…`"):
                                fh_rows = [
                                    {"Field": k, "SHA-256": v}
                                    for k, v in ah.get("field_hashes", {}).items()
                                ]
                                st.dataframe(pd.DataFrame(fh_rows), use_container_width=True, hide_index=True)

                    st.markdown("**Full manifest (raw JSON)**")
                    st.json(hashes)


# ════════════════════════════════════════════════
#  PAGE: ETL AUDIT LOG
# ════════════════════════════════════════════════
elif nav == "📋 ETL Audit Log":
    st.markdown("<h2 style='color:#e8f0fe;font-family:Space Mono,monospace;'>ETL Audit Log</h2>", unsafe_allow_html=True)

    logs = get_audit_logs(db, limit=50)
    if logs:
        rows = []
        for log in logs:
            rows.append({
                "Run Date":   log.get("run_date","")[:19],
                "Batch ID":   log.get("batch_id","")[:8] + "…",
                "Status":     log.get("status",""),
                "Processed":  log.get("records_processed",0),
                "Inserted":   log.get("records_inserted",0),
                "Updated":    log.get("records_updated",0),
                "Archived":   log.get("records_archived",0),
                "Duration(s)":log.get("duration_seconds",0),
                "Errors":     len(log.get("errors") or []),
            })
        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, hide_index=True,
                     column_config={
                         "Status": st.column_config.TextColumn("Status"),
                         "Duration(s)": st.column_config.NumberColumn("Duration(s)", format="%.2f"),
                     })

        # Summary chart
        df_log["Run Date"] = pd.to_datetime(df_log["Run Date"])
        fig = go.Figure()
        for col, color in [("Inserted","#34d399"), ("Updated","#3b82f6"), ("Archived","#f59e0b")]:
            fig.add_trace(go.Bar(name=col, x=df_log["Run Date"], y=df_log[col],
                                 marker_color=color))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                          title="ETL Operations Over Time",
                          xaxis_title="Run Date", yaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ETL runs recorded yet. Run `python etl/pipeline.py` to populate.")
