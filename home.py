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
                        st.session_state.active_tab = "details"
                        st.experimental_rerun()

# -------------------------
# Document Details view
# -------------------------
if tab == "Document Details":
    # Keep session state consistent with the radio
    st.session_state.active_tab = "details"

    st.header("Document Details")

    if st.session_state.selected_doc is None:
        st.warning("No document selected. Select a document in Documents List (use sidebar → Documents List).")
    else:
        # fetch the selected document
        sel_id = st.session_state.selected_doc
        row_df = docs[docs["doc_id"] == sel_id]

        if row_df.empty:
            st.error(f"Selected document id {sel_id} not found.")
        else:
            row = row_df.iloc[0]
            # Two-column layout for details + actions
            left, right = st.columns([3, 1])

            left.subheader(f"{row['name']} (ID: {int(row['doc_id'])})")
            left.markdown(f"**Status:** {row['status']}")
            left.markdown(f"**Owner:** {row['owner']}")
            left.markdown(f"**Created:** {row['created']}")
            left.markdown("**Summary**")
            left.write(row["summary"])

            # Example additional details (replace with real fields)
            left.markdown("---")
            left.markdown("**Metadata**")
            md_table = pd.DataFrame({
                "field": ["doc_id", "name", "status", "owner", "created"],
                "value": [int(row['doc_id']), row['name'], row['status'], row['owner'], row['created']]
            })
            left.table(md_table)

            # Actions
            right.markdown("## Actions")
            if right.button("Back to list"):
                st.session_state.active_tab = "list"
                st.experimental_rerun()

            # placeholders: download/view/annotate
            right.button("Download (placeholder)")
            right.button("Open in external viewer (placeholder)")

# -------------------------
# Footer / diagnostics (optional)
# -------------------------
st.markdown("---")
st.caption("Tip: use the sidebar to search and filter documents. Clicking 'View Details' navigates to the details view.")