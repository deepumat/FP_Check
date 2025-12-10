# Home.py
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from urllib.parse import urlencode

st.set_page_config(page_title="Documents - Home", layout="wide")

# Data - replace with your real source
df = pd.DataFrame({
    "id": [101, 102, 103, 104],
    "title": ["Contract_A.pdf", "Invoice_B.pdf", "Report_C.pdf", "Specs_D.pdf"],
    "owner": ["Alice", "Bob", "Carol", "Dave"],
    "modified": ["2025-11-20", "2025-11-25", "2025-12-01", "2025-12-06"]
})

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None

def open_doc(doc_id: int):
    """Set session state + query param and attempt a JS redirect.
    If JS is blocked, the visible fallback link below allows manual click.
    """
    # 1) primary: set session state
    st.session_state.selected_doc_id = int(doc_id)

    # 2) defensive: set query params (so details page can recover on full reload)
    params = {"doc": str(doc_id)}
    st.experimental_set_query_params(**params)

    # 3) attempt JS redirect using components.html (more reliable than st.markdown JS)
    #    We build a URL that keeps the same pathname but appends the query string.
    target = "?" + urlencode(params)
    js = f"""
    <script>
      try {{
        // attempt to change the URL (this will not reload the page by itself)
        const newUrl = window.location.origin + window.location.pathname + "{target}";
        // Some Streamlit deployments require a full reload to switch pages.
        // Attempt to navigate to the new URL (this reloads the app and lands on the page
        // which will read the 'doc' query param).
        window.location.href = newUrl;
      }} catch(e) {{
        // swallow errors; fallback UI will show link
        console.log('redirect failed', e);
      }}
    </script>
    """
    # Use components.html to ensure the browser runs the script immediately
    components.html(js, height=0, width=0)

    # Defensive rerun (in case JS doesn't execute or is blocked)
    st.experimental_rerun()


st.title("Documents")
st.write("Click **Open** to view details on the Document Details page.")

# Show table for context
st.dataframe(df.set_index("id"), use_container_width=True)

st.write("")  # spacing

buttons_col, info_col = st.columns([1, 6])

with buttons_col:
    for doc_id in df["id"]:
        if st.button("Open", key=f"open_{doc_id}"):
            open_doc(doc_id)

with info_col:
    for _, row in df.iterrows():
        st.markdown(f"**{row['title']}** — {row['owner']} — {row['modified']}")

st.markdown("---")
sel = st.session_state.get("selected_doc_id")
if sel:
    st.info(f"Selected document id: {sel}")
else:
    st.info("No document selected yet.")

# Visible fallback: instructive clickable link (if JS is blocked)
st.markdown(
    """
    **If clicking Open does not navigate automatically:**\n
    - Try the visible link next to a document (opens same page) or refresh the browser.
    - If you want immediate navigation without a reload, consider using AgGrid row-click (I can provide that).
    """
)
# Show explicit manual links for each doc (visible fallback)
for _, row in df.iterrows():
    doc_id = row["id"]
    link = f"[Open {row['title']}](?doc={doc_id})"
    st.write(link)