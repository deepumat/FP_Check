# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Docs (Single-file navigation)", layout="wide")

# -----------------------
# Sample data (replace)
# -----------------------
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

# -----------------------
# Session state defaults
# -----------------------
if "page" not in st.session_state:
    # possible values: "list", "details", "other"
    st.session_state.page = "list"

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None

# -----------------------
# Helper navigation functions
# -----------------------
def go_to_list():
    st.session_state.selected_doc_id = None
    st.session_state.page = "list"

def open_doc(doc_id: int):
    st.session_state.selected_doc_id = int(doc_id)
    st.session_state.page = "details"

# -----------------------
# Sidebar (optional)
# -----------------------
with st.sidebar:
    st.title("Navigation")
    # show current "page" in the sidebar - optional visible control for debugging
    choice = st.radio(
        "Page",
        options=["list", "details", "other"],
        index=["list", "details", "other"].index(st.session_state.page),
        on_change=lambda: setattr(st.session_state, "page", st.session_state.get("page_radio", "list"))
    )
    # keep the radio in sync with session_state.page (readonly style)
    # We don't expect users to change this radio; it's for visibility.
    st.markdown("---")
    st.write("Use the app UI to navigate (Open / Back buttons).")

# -----------------------
# Main rendering logic
# -----------------------
def render_list_view():
    st.header("Documents")
    st.write("Click **Open** to view a document's details below.")

    # Native table for visuals
    st.dataframe(df.set_index("id"), use_container_width=True)

    st.write("")  # spacing

    # Layout: buttons column + info column
    buttons_col, info_col = st.columns([1, 7])

    with buttons_col:
        for doc_id in df["id"]:
            if st.button("Open", key=f"open_{doc_id}"):
                open_doc(doc_id)

    with info_col:
        for _, row in df.iterrows():
            st.markdown(f"**{row['title']}** — {row['owner']} — {row['modified']}")

    st.divider()
    sel = st.session_state.get("selected_doc_id")
    if sel:
        st.info(f"Selected document id (in session): {sel}")
    else:
        st.info("No document selected yet.")

def render_details_view():
    sel_id = st.session_state.get("selected_doc_id")
    st.header("Document Details")

    if sel_id is None:
        st.warning("No document selected. Use the Documents page to open one.")
        if st.button("Back to list"):
            go_to_list()
    else:
        row = df[df["id"] == sel_id]
        if row.empty:
            st.error(f"Selected document id {sel_id} not found.")
            if st.button("Back to list"):
                go_to_list()
        else:
            row = row.iloc[0]
            st.subheader(row["title"])
            st.markdown(f"**Owner:** {row['owner']}")
            st.markdown(f"**Last modified:** {row['modified']}")
            st.markdown("---")
            st.subheader("Summary")
            st.write(row["summary"])
            st.markdown("---")

            # Example actions
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Back to list", key="back_button"):
                    go_to_list()
            with col2:
                if st.button("Open in new tab (fallback)", key="open_new_tab"):
                    # Open details in a new browser tab by creating a URL with a query param,
                    # but note: some hosts or browsers may block JS. This is an optional convenience.
                    params = f"?doc={sel_id}"
                    url = st.runtime.scriptrunner.get_script_run_ctx().session_id if False else None
                    st.write("Use the 'Back to list' button for reliable navigation.")

def render_other_view():
    st.header("Other")
    st.write("Other app content goes here.")
    if st.button("Back to list"):
        go_to_list()

# Choose which view to render
if st.session_state.page == "list":
    render_list_view()
elif st.session_state.page == "details":
    render_details_view()
else:
    render_other_view()