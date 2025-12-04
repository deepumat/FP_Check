import ast
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


# Color code for node types
TAG_COLOR_MAP = {
    "BaseAgreement": "green",
    "Executable": "orange",
    "Supplier": "blue",
    "Clause": "purple",
    "SupportingDocument": "brown",
}


@st.cache_resource
def init_connection_pool() -> ConnectionPool:
    config = Config()
    config.max_connection_pool_size = MAX_CONNECTIONS

    pool = ConnectionPool()
    ok = pool.init([(NEBULA_HOST, NEBULA_PORT)], config)

    if not ok:
        raise RuntimeError("Failed to initialize Nebula Graph connection pool")

    return pool


def get_session(pool: ConnectionPool) -> Session:
    session = pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
    if not session:
        raise RuntimeError("Failed to create Nebula Graph session")
    return session


def execute_query(session: Session, query: str) -> ResultSet:
    use_space = f"USE {NEBULA_SPACE};"
    resp = session.execute(use_space)
    if not resp.is_succeeded():
        raise RuntimeError(resp.error_msg())

    resp = session.execute(query)
    if not resp.is_succeeded():
        raise RuntimeError(resp.error_msg())

    return resp


def resultset_to_df(resp: ResultSet) -> pd.DataFrame:
    col_names = resp.keys()
    rows = []

    for row in resp.rows():
        values = []
        for col in col_names:
            v = row.as_value(col)
            values.append(str(v))
        rows.append(values)

    return pd.DataFrame(rows, columns=col_names)


def parse_tag_list(tag_str: str):
    if tag_str is None:
        return []
    try:
        parsed = ast.literal_eval(tag_str)
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
        return [str(parsed)]
    except:
        s = tag_str.strip().strip("[]")
        if not s:
            return []
        return [x.strip().strip('"').strip("'") for x in s.split(",")]


def pick_color(tags: list[str], is_center=False):
    if is_center:
        return "red"

    for t in tags:
        if t in TAG_COLOR_MAP:
            return TAG_COLOR_MAP[t]
    return "gray"


def visualize_graph(df_conn: pd.DataFrame, df_start: pd.DataFrame):
    if df_conn.empty:
        st.info("No graph data to visualize.")
        return

    center_vid = str(df_start.iloc[0]["vid"])
    center_tags = parse_tag_list(df_start.iloc[0]["tags"])

    net = Network(height="650px", width="100%", directed=True)

    # Add the center node first
    net.add_node(
        center_vid,
        label=center_vid,
        color="red",
        size=30,
        title=f"Center Node: {center_vid}\nTags: {center_tags}",
    )

    # Add all edges and nodes
    for _, row in df_conn.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])
        edge_type = str(row["edge_type"])

        src_tags = parse_tag_list(row["src_tags"])
        dst_tags = parse_tag_list(row["dst_tags"])

        # Add src node (skip if center)
        if src != center_vid:
            net.add_node(
                src,
                label=src,
                color=pick_color(src_tags),
                size=20,
                title=f"{src}\nTags: {src_tags}",
            )

        # Add dst node (skip if center)
        if dst != center_vid:
            net.add_node(
                dst,
                label=dst,
                color=pick_color(dst_tags),
                size=20,
                title=f"{dst}\nTags: {dst_tags}",
            )

        # Add edges
        net.add_edge(src, dst, title=edge_type)

    net.toggle_physics(True)

    html = "graph.html"
    net.save_graph(html)
    with open(html, "r", encoding="utf-8") as f:
        components.html(f.read(), height=650, scrolling=True)


# -------------------------------------------------------
# STREAMLIT APPLICATION
# -------------------------------------------------------

def main():
    st.title("Nebula Graph – Contract Explorer")

    try:
        pool = init_connection_pool()
        session = get_session(pool)
        st.success("Connected to Nebula Graph successfully.")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return

    st.subheader("Node Lookup")
    node_name = st.text_input("Node name (property `name`):", placeholder="Supplier A")

    if st.button("Run query"):
        if not node_name.strip():
            st.warning("Please enter a node name.")
            return

        safe_name = node_name.replace('"', '\\"')

        # Query for the selected node
        q1 = f"""
        MATCH (v)
        WHERE v.name == "{safe_name}"
        RETURN id(v) AS vid, labels(v) AS tags, properties(v) AS props;
        """

        # Query for connections
        q2 = f"""
        MATCH (v)-[e]-(n)
        WHERE v.name == "{safe_name}"
        RETURN
            id(v) AS src, labels(v) AS src_tags, properties(v) AS src_props,
            id(n) AS dst, labels(n) AS dst_tags, properties(n) AS dst_props,
            type(e) AS edge_type, e AS edge_props;
        """

        try:
            df_start = resultset_to_df(execute_query(session, q1))
            df_conn_raw = resultset_to_df(execute_query(session, q2))

            if df_start.empty:
                st.error("No such node found.")
                return

            st.session_state["df_start"] = df_start
            st.session_state["df_conn_raw"] = df_conn_raw

        except Exception as e:
            st.error(f"Query error: {e}")
            return

    # ---------------------------------------------------
    # Display + Filters + Graph (After Query Runs)
    # ---------------------------------------------------

    if "df_start" in st.session_state:
        df_start = st.session_state["df_start"]
        df_conn_raw = st.session_state["df_conn_raw"]

        st.markdown("### Node Details")
        st.dataframe(df_start)

        if df_conn_raw.empty:
            st.info("This node has no connected edges.")
            return

        # Build available lists
        edge_types = sorted(df_conn_raw["edge_type"].unique())

        tag_set = set()
        for col in ["src_tags", "dst_tags"]:
            for t in df_conn_raw[col].dropna():
                tag_set.update(parse_tag_list(t))
        available_tags = sorted(tag_set)

        # FILTER UI
        st.subheader("Filters")

        selected_edge_types = st.multiselect(
            "Edge Types",
            options=edge_types,
            default=edge_types,    # all selected by default
        )

        selected_tags = st.multiselect(
            "Vertex Tags",
            options=available_tags,
            default=["BaseAgreement", "Executable"],   # PRESELECTED TAGS
        )

        # Apply filters
        df_conn = df_conn_raw.copy()

        # filter edge types
        df_conn = df_conn[df_conn["edge_type"].isin(selected_edge_types)]

        # filter tags
        def node_matches(tag_str):
            tags = parse_tag_list(tag_str)
            return any(t in selected_tags for t in tags)

        df_conn = df_conn[
            df_conn["src_tags"].apply(node_matches)
            | df_conn["dst_tags"].apply(node_matches)
        ]

        st.markdown("### Filtered Connections")
        st.dataframe(df_conn)

        st.markdown("### Graph Visualization")
        visualize_graph(df_conn, df_start)


if __name__ == "__main__":
    main()