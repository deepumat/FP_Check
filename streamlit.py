import streamlit as st
import pandas as pd

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from nebula3.gclient.net.Session import Session
from nebula3.data.ResultSet import ResultSet

from pyvis.network import Network
import streamlit.components.v1 as components


# -------------------------------------------------------
# HARD-CODED NEBULA GRAPH CONFIG
# -------------------------------------------------------
NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
NEBULA_SPACE = "contracts_space"   # change to your space name
MAX_CONNECTIONS = 10
# -------------------------------------------------------


@st.cache_resource
def init_connection_pool() -> ConnectionPool:
    """
    Create a Nebula Graph connection pool using hard-coded values.
    """
    config = Config()
    config.max_connection_pool_size = MAX_CONNECTIONS

    pool = ConnectionPool()
    ok = pool.init([(NEBULA_HOST, NEBULA_PORT)], config)

    if not ok:
        raise RuntimeError("Failed to initialize Nebula Graph connection pool")

    return pool


def get_session(pool: ConnectionPool) -> Session:
    """
    Get a session using hard-coded user + password.
    """
    session = pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
    if not session:
        raise RuntimeError("Failed to create Nebula Graph session")
    return session


def execute_query(session: Session, query: str) -> ResultSet:
    """
    Execute nGQL inside hard-coded SPACE.
    """
    use_space = f"USE {NEBULA_SPACE};"
    resp = session.execute(use_space)
    if not resp.is_succeeded():
        raise RuntimeError(f"Failed to USE space {NEBULA_SPACE}: {resp.error_msg()}")

    resp = session.execute(query)
    if not resp.is_succeeded():
        raise RuntimeError(f"Query failed: {resp.error_msg()}")
    return resp


def resultset_to_df(resp: ResultSet) -> pd.DataFrame:
    """
    Convert ResultSet to DataFrame.
    NOTE: We cast values to string for display/filters.
    """
    col_names = resp.keys()
    rows = []

    for row in resp.rows():
        values = []
        for col in col_names:
            v = row.as_value(col)
            values.append(str(v))
        rows.append(values)

    return pd.DataFrame(rows, columns=col_names)


def visualize_graph(conn_df: pd.DataFrame, center_vid: str | None = None):
    """
    Build and display an interactive graph using pyvis in Streamlit.

    conn_df must contain at least columns:
      - src
      - dst
      - edge_type (optional, used as tooltip)
    """
    if conn_df.empty:
        st.info("No connections to visualize (after filters).")
        return

    net = Network(height="600px", width="100%", directed=True, notebook=False)

    # Add nodes and edges
    for _, row in conn_df.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        edge_type = str(row["edge_type"]) if "edge_type" in conn_df.columns else ""

        # Add / update src node
        if center_vid is not None and src == str(center_vid):
            net.add_node(
                src,
                label=src,
                color="red",
                size=25,
                title=f"Center node: {src}",
            )
        else:
            net.add_node(src, label=src)

        # Add / update dst node
        if center_vid is not None and dst == str(center_vid):
            net.add_node(
                dst,
                label=dst,
                color="red",
                size=25,
                title=f"Center node: {dst}",
            )
        else:
            net.add_node(dst, label=dst)

        # Add edge
        net.add_edge(src, dst, title=edge_type)

    net.toggle_physics(True)

    # Save and render inside Streamlit
    html_file = "graph.html"
    net.save_graph(html_file)
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=600, scrolling=True)


# -------------------------------------------------------
# STREAMLIT APPLICATION
# -------------------------------------------------------

def main():
    st.title("Nebula Graph – Node Connections Explorer")

    # Connect once (hard-coded)
    try:
        pool = init_connection_pool()
        session = get_session(pool)
        st.success(
            f"Connected to Nebula Graph at {NEBULA_HOST}:{NEBULA_PORT}, "
            f"space `{NEBULA_SPACE}`."
        )
    except Exception as e:
        st.error(f"Failed to connect to Nebula Graph: {e}")
        return

    st.subheader("Search connections for a node")

    node_name = st.text_input(
        "Node name (value of property `name`):",
        placeholder="e.g., Supplier A, Base Agreement 123",
    )

    st.markdown("#### Optional filters")

    # User input: edge type filtering
    edge_type_filter = st.text_input(
        "Edge types to include (comma-separated, blank = all):",
        placeholder="e.g., ExecutableEdge, SupplierOf, ContainsClause",
    )

    # User input: vertex tag filtering for connected nodes
    tag_filter = st.text_input(
        "Vertex tags for connected nodes (comma-separated, blank = all):",
        placeholder="e.g., BaseAgreement, Executable, Supplier, Clause",
    )

    if st.button("Get connections"):
        if not node_name.strip():
            st.warning("Please enter a node name.")
            return

        safe_name = node_name.replace('"', '\\"')

        # Query 1: starting node details
        start_node_query = f"""
        MATCH (v)
        WHERE v.name == "{safe_name}"
        RETURN
            id(v) AS vid,
            labels(v) AS tags,
            properties(v) AS props;
        """

        # Query 2: all neighbors + edges
        # We get everything first, then filter in Python based on the user inputs.
        connections_query = f"""
        MATCH (v)-[e]-(n)
        WHERE v.name == "{safe_name}"
        RETURN
            id(v) AS src,
            labels(v) AS src_tags,
            properties(v) AS src_props,
            id(n) AS dst,
            labels(n) AS dst_tags,
            properties(n) AS dst_props,
            type(e) AS edge_type,
            e AS edge_props;
        """

        try:
            # Run both queries
            start_resp = execute_query(session, start_node_query)
            conn_resp = execute_query(session, connections_query)

            with st.expander("Show executed nGQL"):
                st.code(start_node_query.strip(), language="sql")
                st.code(connections_query.strip(), language="sql")

            # Node details
            st.markdown("### Node details")
            if start_resp.is_empty():
                st.error("No node found with this `name`.")
                return

            df_start = resultset_to_df(start_resp)
            st.dataframe(df_start, use_container_width=True)

            # Determine center VID from first row
            center_vid = str(df_start.iloc[0]["vid"])

            # Connections
            if conn_resp.is_empty():
                st.info("Node exists but has no connected edges.")
                return

            df_conn = resultset_to_df(conn_resp)

            # ---------------------------------------------------
            # Apply edge-type filter if provided
            # ---------------------------------------------------
            if edge_type_filter.strip():
                edge_types = [
                    e.strip()
                    for e in edge_type_filter.split(",")
                    if e.strip()
                ]
                df_conn = df_conn[df_conn["edge_type"].isin(edge_types)]

            # ---------------------------------------------------
            # Apply tag filter if provided (on src_tags OR dst_tags)
            # df_conn["src_tags"] / ["dst_tags"] are strings like: ["Supplier"]
            # We just check substring membership for simplicity.
            # ---------------------------------------------------
            if tag_filter.strip():
                tags = [
                    t.strip()
                    for t in tag_filter.split(",")
                    if t.strip()
                ]

                def has_any_tag(tag_str: str) -> bool:
                    return any(t in tag_str for t in tags)

                df_conn = df_conn[
                    df_conn["src_tags"].apply(has_any_tag)
                    | df_conn["dst_tags"].apply(has_any_tag)
                ]

            st.markdown("### Connected nodes and edges (after filters)")
            if df_conn.empty:
                st.info("No connections match the specified filters.")
                return

            st.dataframe(df_conn, use_container_width=True)

            # ---------------------------------------------------
            # Graph visualization (for filtered connections)
            # ---------------------------------------------------
            st.markdown("### Graph visualization")
            visualize_graph(df_conn, center_vid=center_vid)

        except Exception as e:
            st.error(f"Error running query: {e}")


if __name__ == "__main__":
    main()