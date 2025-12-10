# pages/Document Details.py
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="Document Details")

# Sample data (same source as Home.py)
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

def get_selected_doc_id():
    # 1) preferred: session_state
    if st.session_state.get("selected_doc_id") is not None:
        return int(st.session_state.selected_doc_id)

    # 2) fallback: query params
    params = st.experimental_get_query_params()
    doc_vals = params.get("doc")
    if doc_vals:
        try:
            return int(doc_vals[0])
        except (ValueError, TypeError):
            return None

    return None

st.title("Document Details")

sel_id = get_selected_doc_id()

if sel_id is None:
    st.warning("No document selected. Select from Home or click a visible link in Home.")
    st.write("You can also manually append `?doc=<id>` to the app URL.")
else:
    row = df[df["id"] == sel_id]
    if row.empty:
        st.error(f"Selected document id {sel_id} not found.")
    else:
        row = row.iloc[0]
        st.header(row["title"])
        st.markdown(f"**Owner:** {row['owner']}")
        st.markdown(f"**Last modified:** {row['modified']}")
        st.markdown("---")
        st.subheader("Summary")
        st.write(row["summary"])

        st.markdown("---")
        if st.button("Back to list"):
            # clear selection and navigate to Home root
            st.session_state.selected_doc_id = None
            st.experimental_set_query_params()  # clear params
            # Attempt JS redirect back to root
            js = "<script>window.location.href = window.location.origin + window.location.pathname;</script>"
            components.html(js, height=0, width=0)
            st.experimental_rerun()