import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import networkx as nx
import random
from ortools.graph import pywrapgraph
from .utility import print_info, setup, optimize, grid_setup, add_party_preference, create_district_map

# Copy your existing functions here: print_info, setup, optimize, grid_setup, add_party_preference

app = dash.Dash(__name__)
server = app.server  # Needed for Heroku deployment

app.layout = html.Div([
    html.H1("District Visualization"),
    html.Div([
        html.Label("Grid Size"),
        dcc.Slider(id='grid-size-slider', min=5, max=20, step=1, value=10, marks={i: str(i) for i in range(5, 21, 5)}),
    ]),
    html.Div([
        html.Label("Districts"),
        dcc.Slider(id='districts-slider', min=2, max=10, step=1, value=5, marks={i: str(i) for i in range(2, 11, 2)}),
    ]),
    html.Div([
        html.Label("P0"),
        dcc.Slider(id='p0-slider', min=0.1, max=0.9, step=0.1, value=0.5, marks={i/10: str(i/10) for i in range(1, 10)}),
    ]),
    html.Div([
        html.Label("C"),
        dcc.Slider(id='c-slider', min=1, max=10, step=1, value=3, marks={i: str(i) for i in range(1, 11)}),
    ]),
    html.Div([
        html.Label("R"),
        dcc.Slider(id='r-slider', min=1, max=10, step=1, value=3, marks={i: str(i) for i in range(1, 11)}),
    ]),
    html.Div([
        html.Label("N"),
        dcc.Slider(id='n-slider', min=1, max=10, step=1, value=5, marks={i: str(i) for i in range(1, 11)}),
    ]),
    dcc.Graph(id='district-map'),
    dcc.Graph(id='district-association-map')
])

@app.callback(
    [Output('district-map', 'figure'),
     Output('district-association-map', 'figure')],
    [Input('grid-size-slider', 'value'),
     Input('districts-slider', 'value'),
     Input('p0-slider', 'value'),
     Input('c-slider', 'value'),
     Input('r-slider', 'value'),
     Input('n-slider', 'value')]
)
def update_graphs(grid_size, districts, p_0, c, r, n):
    # Your existing update_plots function logic goes here
    # Make sure to return two figures instead of updating fig_widget and fig_widget2
    
    # Example (you'll need to adapt this to your specific logic):
    districts, grid_size, district_centers, pops, grid = grid_setup(grid_size, districts, n, p_0, c, r)
    start_nodes, end_nodes, capacities, costs, supplies, source, sink, pops, total_pop = setup(districts, grid_size, district_centers, pops)
    block_assignments = optimize(start_nodes, end_nodes, capacities, costs, supplies, source, sink, grid, grid_size)

    x_coords, y_coords, colors, district_colors, party_counts = create_district_map(grid, block_assignments)

    fig1 = go.Figure(data=go.Scatter(
        x=x_coords, y=y_coords, mode='markers',
        marker=dict(color=colors, size=10),
        text=[f'District {grid.nodes[node]["district"]}' for node in grid.nodes()],
        hoverinfo='text'
    ))

    fig2 = go.Figure(data=go.Scatter(
        x=x_coords, y=y_coords, mode='markers',
        marker=dict(color=district_colors, size=10),
        text=[f'District {grid.nodes[node]["district"]}' for node in grid.nodes()],
        hoverinfo='text'
    ))

    fig1.update_layout(
        title=f'District Map (Blue: {party_counts["blue"]}, Red: {party_counts["red"]})',
        xaxis_title='X', yaxis_title='Y',
        xaxis=dict(range=[-1, grid_size]), yaxis=dict(range=[-1, grid_size])
    )

    fig2.update_layout(
        title='District Association Map',
        xaxis_title='X', yaxis_title='Y',
        xaxis=dict(range=[-1, grid_size]), yaxis=dict(range=[-1, grid_size])
    )

    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True)