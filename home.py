# Home.py
import streamlit as st
import pandas as pd
from urllib.parse import urlencode

st.set_page_config(page_title="Documents - Home", layout="wide")

# ---------------------------
# Session state initialization
# ---------------------------
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None

# ---------------------------
# Replace this with your real data source
# ---------------------------
df = pd.DataFrame({
    "id": [101, 102, 103, 104],
    "title": ["Contract_A.pdf", "Invoice_B.pdf", "Report_C.pdf", "Specs_D.pdf"],
    "owner": ["Alice", "Bob", "Carol", "Dave"],
    "modified": ["2025-11-20", "2025-11-25", "2025-12-01", "2025-12-06"]
})
df_display = df.set_index("id")

# ---------------------------
# Helper: open a document (set state + navigate)
# ---------------------------
def open_doc(doc_id: int):
    """
    Save selected doc to session_state and navigate to 'Document Details' page.
    We set query params (doc) as a fallback so the details page can pick it up
    even if session_state is lost. The JS redirect attempts to switch pages
    immediately in the browser.
    """
    st.session_state.selected_doc_id = int(doc_id)

    # Also set query params so the details page can still read the selection.
    # Setting both 'page' and 'doc' here is defensive.
    params = {"page": "Document Details", "doc": str(doc_id)}
    st.experimental_set_query_params(**params)

    # JavaScript redirect to attempt to switch pages immediately.
    # This works in most Streamlit deployments; it's a pragmatic and commonly-used approach.
    redirect_url = (
        "window.location.href = window.location.origin + "
        "window.location.pathname + '?' + decodeURIComponent('{}');"
    ).format(urlencode(params))
    js = f"<script>{redirect_url}</script>"

    st.markdown(js, unsafe_allow_html=True)

    # Defensive rerun (in case the JS doesn't run immediately for some reason).
    st.experimental_rerun()


# ---------------------------
# UI: Document list page
# ---------------------------
st.title("Documents")
st.write("Click **Open** to view details on the Document Details page.")

# Show the table for context (native)
st.dataframe(df_display, use_container_width=True)

st.write("")  # spacing

# Buttons + compact summary column layout
buttons_col, info_col = st.columns([1, 6])

with buttons_col:
    for doc_id in df["id"]:
        # stable key per doc so Streamlit keeps button state consistent
        if st.button("Open", key=f"open_{doc_id}"):
            open_doc(doc_id)

with info_col:
    for _, row in df.iterrows():
        st.markdown(f"**{row['title']}** — {row['owner']} — {row['modified']}")

# Optional: show currently selected doc
st.divider()
sel = st.session_state.get("selected_doc_id")
if sel is not None:
    st.info(f"Currently selected document id: {sel}")
else:
    st.info("No document selected yet.")