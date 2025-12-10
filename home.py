# streamlit_app.py
import streamlit as st
import pandas as pd

# -------------------------
# Sample data (replace with your source)
# -------------------------
docs = pd.DataFrame([
    {"doc_id": 1, "name": "Agreement A", "status": "Active",  "owner": "Alice",   "created": "2025-01-10", "summary": "Master services agreement."},
    {"doc_id": 2, "name": "Invoice B",   "status": "Pending", "owner": "Bob",     "created": "2025-03-22", "summary": "Invoice for project X."},
    {"doc_id": 3, "name": "Contract C",  "status": "Expired", "owner": "Carol",   "created": "2024-11-05", "summary": "Expired supplier contract."},
    {"doc_id": 4, "name": "NDA D",       "status": "Active",  "owner": "Dave",    "created": "2025-07-07", "summary": "Mutual NDA."},
    {"doc_id": 5, "name": "SLA E",       "status": "Pending", "owner": "Eve",     "created": "2025-09-29", "summary": "Service level agreement."},
])

# -------------------------
# Session state initialization
# -------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "list"   # values: "list" or "details"

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# -------------------------
# Sidebar: navigation + filters
# -------------------------
st.sidebar.title("Navigation & Filters")

tab = st.sidebar.radio(
    "Navigate",
    ["Documents List", "Document Details"],
    index=0 if st.session_state.active_tab == "list" else 1
)

# Search and filters
st.sidebar.markdown("---")
search_text = st.sidebar.text_input("Search (id or name)", value="")
all_statuses = sorted(docs["status"].unique().tolist())
status_selection = st.sidebar.multiselect("Status (filter)", options=all_statuses, default=all_statuses)

# Optional: sort order
sort_by = st.sidebar.selectbox("Sort by", ["doc_id", "name", "created"], index=0)
sort_asc = st.sidebar.checkbox("Ascending", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Quick actions")
if st.sidebar.button("Clear selection"):
    st.session_state.selected_doc = None
    st.session_state.active_tab = "list"
    st.experimental_rerun()

# -------------------------
# Apply filters to dataframe
# -------------------------
def filter_docs(df, search, statuses):
    df_filtered = df.copy()
    if statuses:
        df_filtered = df_filtered[df_filtered["status"].isin(statuses)]
    if search:
        s = str(search).strip().lower()
        # match id or name (partial)
        df_filtered = df_filtered[
            df_filtered["name"].str.lower().str.contains(s) |
            df_filtered["doc_id"].astype(str).str.contains(s)
        ]
    return df_filtered

filtered = filter_docs(docs, search_text, status_selection)
filtered = filtered.sort_values(by=sort_by, ascending=sort_asc).reset_index(drop=True)

# -------------------------
# Page title
# -------------------------
st.set_page_config(page_title="Document Manager", layout="wide")
st.title("Document Manager")

# -------------------------
# Documents List view
# -------------------------
if tab == "Documents List":
    # Ensure session state tab is consistent with the radio
    st.session_state.active_tab = "list"

    st.header("All Documents")
    st.write(f"Showing **{len(filtered)}** documents (filtered) — use the sidebar to refine.")

    if filtered.empty:
        st.info("No documents match the current filter/search criteria.")
    else:
        # Group by status and render each group inside an expander
        for status in filtered["status"].unique():
            group = filtered[filtered["status"] == status]
            with st.expander(f"{status} — {len(group)}", expanded=True):
                for _, row in group.iterrows():
                    cols = st.columns([4, 1, 1, 1])
                    # main info column
                    cols[0].markdown(
                        f"**{int(row['doc_id'])} — {row['name']}**  \n"
                        f"*Owner:* {row['owner']}  •  *Created:* {row['created']}  \n\n"
                        f"{row['summary']}"
                    )
                    # small metadata columns
                    cols[1].write(row["status"])
                    cols[2].write(row["created"])
                    # View Details button
                    view_key = f"view_{int(row['doc_id'])}"
                    if cols[3].button("View Details", key=view_key):
                        st.session_state.selected_doc = int(row["doc_id"])
                        st