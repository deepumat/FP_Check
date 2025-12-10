import streamlit as st
import pandas as pd

# Sample data
docs = pd.DataFrame({
    "doc_id": [1, 2, 3],
    "name": ["Agreement A", "Invoice B", "Contract C"],
    "status": ["Active", "Pending", "Expired"]
})

# Session state initialization
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "list"

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# -------------------------------------------------------------------
# TAB CONTROLLER
# -------------------------------------------------------------------
tab = st.radio(
    "Navigation",
    ["Documents List", "Document Details"],
    index=0 if st.session_state.active_tab == "list" else 1
)

# -------------------------------------------------------------------
# TAB 1 – DOCUMENT LIST
# -------------------------------------------------------------------
if tab == "Documents List":

    st.header("All Documents")

    # Display table and clickable buttons
    for i, row in docs.iterrows():
        cols = st.columns([4, 2])
        cols[0].write(f"{row['doc_id']} – {row['name']} ({row['status']})")

        if cols[1].button("View Details", key=f"view_{row['doc_id']}"):
            st.session_state.selected_doc = int(row['doc_id'])
            st.session_state.active_tab = "details"
            st.rerun()

# -------------------------------------------------------------------
# TAB 2 – DOCUMENT DETAILS
# -------------------------------------------------------------------
if tab == "Document Details":

    st.header("Document Details")

    if st.session_state.selected_doc is None:
        st.info("Select a document from the list first.")
    else:
        # Fetch the selected row
        row = docs[docs.doc_id == st.session_state.selected_doc].iloc[0]

        st.subheader(row["name"])
        st.write("Document ID:", row["doc_id"])
        st.write("Status:", row["status"])

        # Add more metadata, graphs, or even Nebula Graph visualizations here