import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

st.set_page_config(layout="wide")

# 1. Sample Data (Hierarchical structure)
data = {
    'Project': ['Project A', 'Project B', 'Project C'],
    'Manager': ['Alice', 'Bob', 'Charlie'],
    'Status': ['In Progress', 'Completed', 'On Hold'],
    'Details': [
        [{'Task': 'Design', 'Hours': 40}, {'Task': 'Implement', 'Hours': 120}],
        [{'Task': 'Test', 'Hours': 30}, {'Task': 'Deploy', 'Hours': 10}],
        [{'Task': 'Research', 'Hours': 80}]
    ]
}
df = pd.DataFrame(data)

# 2. JavaScript Code for the Row Expander and Nested Grid
# This JS snippet defines a cell renderer that, when clicked, creates a new AgGrid instance
# within an expander element in the row details.
row_expander_renderer = JsCode("""
class RowExpanderRenderer {
    init(params) {
        this.eGui = document.createElement('div');
        this.eGui.innerHTML = `
            <button style="border: none; background: none; cursor: pointer;">
                +
            </button>
        `;
        this.expanded = false;
        this.detailsDiv = document.createElement('div');
        this.detailsDiv.style.display = 'none';
        this.eGui.appendChild(this.detailsDiv);
        
        this.eGui.querySelector('button').addEventListener('click', () => {
            this.toggleDetails(params.data);
        });
    }

    getGui() {
        return this.eGui;
    }

    toggleDetails(data) {
        this.expanded = !this.expanded;
        if (this.expanded) {
            this.eGui.querySelector('button').textContent = '-';
            this.detailsDiv.style.display = 'block';
            this.renderNestedGrid(data.Details);
        } else {
            this.eGui.querySelector('button').textContent = '+';
            this.detailsDiv.style.display = 'none';
            // Destroy the nested grid instance when collapsing to prevent memory leaks/conflicts
            if (this.nestedGridApi) {
                this.nestedGridApi.destroy();
            }
        }
    }

    renderNestedGrid(detailsData) {
        const gridOptions = {
            columnDefs: [
                { headerName: "Task", field: "Task", flex: 1 },
                { headerName: "Hours", field: "Hours", flex: 1 }
            ],
            rowData: detailsData,
            domLayout: 'autoHeight',
            // Add any other desired grid options for the nested table
        };
        // Use the AgGrid API to create a new grid in the details div
        new agGrid.Grid(this.detailsDiv, gridOptions);
        this.nestedGridApi = gridOptions.api;
    }
}
""")

# 3. Configure the main grid using GridOptionsBuilder
gb = GridOptionsBuilder.from_dataframe(df)

# Configure a specific column to use the custom cell renderer
gb.configure_column(
    "Project",
    cellRenderer=row_expander_renderer,
    onCellClicked=JsCode("function(params) { params.colDef.cellRenderer.toggleDetails(params.data); }").js_code
)

# Configure other columns as needed
gb.configure_default_column(editable=False, filter=True)
gb.configure_selection(selection_mode='single', use_checkbox=True)

grid_options = gb.build()

# 4. Display the AgGrid table
st.subheader("Projects Overview with Expandable Rows")
AgGrid(
    df,
    gridOptions=grid_options,
    height=300,
    allow_unsafe_jscode=True, # Must be True to use JsCode
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    theme='streamlit'
)
