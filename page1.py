# pages/Document Details.py
import streamlit as st

st.set_page_config(page_title="Document Details")

# Helper: try session_state first, fallback to query params
def get_selected_doc_id():
    # 1) session_state
    if st.session_state.get("selected_doc_id") is not None:
        return int(st.session_state["selected_doc_id"])

    # 2) query params fallback
    params = st.experimental_get_query_params()
    doc_vals = params.get("doc")
    if doc_vals:
        try:
            return int(doc_vals[0])
        except (ValueError, TypeError):
            return None

    # nothing found
    return None


# For this demo, re-create the same dataset or load it from your source.
import pandas as pd
df = pd.DataFrame({
    "id": [101, 102, 103, 104],
    "title": ["Contract_A.pdf", "Invoice_B.pdf", "Report_C.pdf", "Specs_D.pdf"],
    "owner": ["Alice", "Bob", "Carol", "Dave"],
    "modified": ["2025-11-20", "2025-11-25", "2025-12-01", "2025-12-06"],
    "summary": [
        "Contract with vendor X — 12 pages.",
        "Invoice for order #4321 — paid.",
        "Quarterly report — contains charts.",
        "Technical specs v2.1 — draft."
    ]
})

st.title("Document Details")

sel_id = get_selected_doc_id()

if sel_id is None:
    st.warning("No document selected. Please select a document from the Home page.")
    st.write("Tip: return to the Home page and click an 'Open' button for a document.")
else:
    row = df[df["id"] == sel_id]
    if row.empty:
        st.error(f"Selected document id {sel_id} not found in the data source.")
    else:
        row = row.iloc[0]
        st.header(row["title"])
        st.markdown(f"**Owner:** {row['owner']}")
        st.markdown(f"**Last modified:** {row['modified']}")
        st.markdown("---")
        st.subheader("Summary")
        st.write(row["summary"])

        st.markdown("---")
        st.write("Actions:")
        if st.button("Back to list"):
            # Clear selection and navigate back to Home (root).
            st.session_state.selected_doc_id = None
            # Clear query params for cleanliness
            st.experimental_set_query_params()
            # JS redirect back to the Home page (root)
            js = "<script>window.location.href = window.location.origin + window.location.pathname;</script>"
            st.markdown(js, unsafe_allow_html=True)
            st.experimental_rerun()