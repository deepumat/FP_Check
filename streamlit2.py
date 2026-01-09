import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

data = [
    {"Title": "BT_Master Technology Agreement", "Contract": "CW7047", "Level": 0},
    {"Title": "Order Form – Global Network & Hardware Maintenance", "Contract": "CW7240", "Level": 1},
    {"Title": "Flexible Cloud License (5 Years)", "Contract": "CW8734", "Level": 1},
    {"Title": "Crossbalance Switch Replacement", "Contract": "CW7590", "Level": 1},
    {"Title": "Cancelled", "Contract": "", "Level": 0},
    {"Title": "Order Form – Hong Kong 2021", "Contract": "CW4003", "Level": 1},
    {"Title": "Order Form – China 2021", "Contract": "CW4004", "Level": 1},
]

df = pd.DataFrame(data)

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_column(
    "Title",
    cellRenderer="""
    function(params) {
        let indent = '&nbsp;'.repeat(params.data.Level * 6);
        return indent + params.value;
    }
    """
)
gb.configure_column("Level", hide=True)
gb.configure_grid_options(domLayout='normal')

grid_options = gb.build()

st.title("Agreement Tree")
AgGrid(
    df,
    gridOptions=grid_options,
    use_container_width=True,
    allow_unsafe_jscode=True
)