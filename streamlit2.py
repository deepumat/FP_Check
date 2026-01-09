import pandas as pd

data = [
    {"id": "BT", "parent": None, "title": "BT_Master Technology Agreement", "contract": "CW7047"},
    {"id": "BT1", "parent": "BT", "title": "Global Network & Hardware Maintenance Renewal 2020", "contract": "CW7240"},
    {"id": "BT2", "parent": "BT", "title": "Flexible Cloud License – 5 Years (Singapore)", "contract": "CW8734"},
    {"id": "BT3", "parent": "BT", "title": "Crossbalance Switch Replacement – Malaysia", "contract": "CW7590"},

    {"id": "CANCEL", "parent": None, "title": "Cancelled", "contract": ""},
    {"id": "HK", "parent": "CANCEL", "title": "MAC Order Form – Hong Kong 2021", "contract": "CW4003"},
    {"id": "CN", "parent": "CANCEL", "title": "MAC Order Form – China 2021", "contract": "CW4004"},
    {"id": "UAE", "parent": "CANCEL", "title": "MAC Order Form – UAE 2021", "contract": "CW4005"},
]

df = pd.DataFrame(data)

def build_path(row, df):
    path = [row["title"]]
    parent = row["parent"]

    while parent:
        parent_row = df[df["id"] == parent].iloc[0]
        path.insert(0, parent_row["title"])
        parent = parent_row["parent"]

    return path

df["path"] = df.apply(lambda r: build_path(r, df), axis=1)

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(layout="wide")
st.title("Agreement Tree")

gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_grid_options(
    treeData=True,
    getDataPath="function(data) { return data.path; }",
    autoGroupColumnDef={
        "headerName": "Agreement Title",
        "cellRendererParams": {
            "suppressCount": True,
        },
        "width": 600,
    },
)

gb.configure_column("contract", headerName="Contract Number", width=150)
gb.configure_column("id", hide=True)
gb.configure_column("parent", hide=True)
gb.configure_column("path", hide=True)

gridOptions = gb.build()

AgGrid(
    df,
    gridOptions=gridOptions,
    use_container_width=True,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=True
)