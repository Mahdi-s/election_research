import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import networkx as nx
import random
import math
import os
from ortools.graph import pywrapgraph
from utility import setup, optimize, grid_setup, create_district_map, find_eg, step_five_finder, refined_step_five_finder, find_winners

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'], title="District Visualization", update_title="Loading...", meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Define styles
CARD_STYLE = {
    'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)',
    'border-radius': '8px',
    'margin-bottom': '20px',
    'background-color': '#222831',  # Dark background color
    'color': 'white',  # White text color
}

SLIDER_STYLE = {
    'margin-bottom': '20px',
    'color': 'white',  # White text color
}

GRAPH_STYLE = {
    'height': '100%',
    'width': '100%',
    'background-color': '#2c3e50',  # Dark background color
}

FONT_STYLE = {
    'font-family': 'Roboto, sans-serif',
    'color': 'white',  # White text color
}

HEADLINE_STYLE = {
    'font-family': 'Roboto, sans-serif',
    'color': 'black',  # Black text color
}

CONTAINER_STYLE = {
    'background-color': '#F3F7EC',  # Match the background color of the boxes
    'padding': '20px'
}

def create_slider_with_tooltip(id, label, min_value, max_value, step, value, tooltip):
    if isinstance(min_value, int) and isinstance(max_value, int):
        marks = {i: str(i) for i in range(min_value, max_value+1, max(1, (max_value-min_value)//5))}
    else:
        num_marks = 5
        marks = {round(min_value + i * (max_value - min_value) / (num_marks - 1), 2): str(round(min_value + i * (max_value - min_value) / (num_marks - 1), 2)) for i in range(num_marks)}
    
    return html.Div([
        dbc.Label([label, dbc.Badge("?", color="info", className="ml-1", id=f"{id}-tooltip")]),
        dcc.Slider(
            id=id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks=marks,
            className="mt-1"
        ),
        dbc.Tooltip(tooltip, target=f"{id}-tooltip"),
    ], style=SLIDER_STYLE)

app.layout = dbc.Container([
    html.H1("Visualization of Impossibility Theorem For Gerrymandering", className="text-center my-4", style=HEADLINE_STYLE),
    
    dbc.Card([
        dbc.CardBody([
            html.H4("About This Visualization", className="card-title", style=FONT_STYLE),
            html.P(
                "Welcome to our interactive web app designed to shed light on the intricate world of gerrymandering! Gerrymandering, the manipulation of electoral district boundaries for political advantage, has been a hot topic in democratic governance for centuries. Our app brings the power of advanced computational methods and mathematical models right to your fingertips. Dive in to explore how different district configurations can impact election results. Adjust various parameters and constraints, and watch as our simulation tool dynamically updates graphs and calculations, providing a clear and engaging way to understand the effects of gerrymandering. Whether you're a policymaker, political analyst, or just a curious citizen, our app offers valuable insights into the fairness and efficiency of electoral processes.",
                style=FONT_STYLE
            ),
        ])
    ], style=CARD_STYLE),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Control Panel", className="card-title", style=FONT_STYLE),
                    create_slider_with_tooltip('grid-size-slider', "Grid Size", 5, 20, 1, 10, "The size of the grid for district visualization"),
                    create_slider_with_tooltip('districts-slider', "Districts", 2, 10, 1, 5, "The number of districts to create"),
                    create_slider_with_tooltip('p0-slider', "P0", 0.1, 0.9, 0.1, 0.5, "Probability parameter for district generation"),
                    create_slider_with_tooltip('c-slider', "C", 1, 10, 1, 3, "Parameter C for district generation"),
                    create_slider_with_tooltip('r-slider', "R", 1, 10, 1, 3, "Parameter R for district generation"),
                    create_slider_with_tooltip('n-slider', "N", 1, 10, 1, 5, "Parameter N for district generation"),
                ])
            ], style=CARD_STYLE),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Calculations", className="card-title", style=FONT_STYLE),
                    html.Div(id='calculations-table', style=FONT_STYLE),
                ])
            ], style=CARD_STYLE),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("District Map", className="card-title", style=FONT_STYLE),
                    html.Div(id='warning-message', className="text-danger font-weight-bold", style=FONT_STYLE),
                    dcc.Graph(id='district-map', style=GRAPH_STYLE),
                ])
            ], style=CARD_STYLE),
            dbc.Card([
                dbc.CardBody([
                    html.H4("District Association Map", className="card-title", style=FONT_STYLE),
                    dcc.Graph(id='district-association-map', style=GRAPH_STYLE),
                ])
            ], style=CARD_STYLE),
        ], md=8),
    ], className="mt-4"),
    
    dcc.Store(id='missed-nodes-store')
], fluid=True, style=CONTAINER_STYLE)

@app.callback(
    [Output('district-map', 'figure'),
     Output('district-association-map', 'figure'),
     Output('warning-message', 'children'),
     Output('missed-nodes-store', 'data'),
     Output('calculations-table', 'children')],
    [Input('grid-size-slider', 'value'),
     Input('districts-slider', 'value'),
     Input('p0-slider', 'value'),
     Input('c-slider', 'value'),
     Input('r-slider', 'value'),
     Input('n-slider', 'value')]
)
def update_graphs(grid_size, districts, p_0, c, r, n):
    try:
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
            title=dict(text=f'District Map (Blue: {party_counts["blue"]}, Red: {party_counts["red"]})', font=dict(size=18)),
            xaxis_title=dict(text='X', font=dict(size=14)),
            yaxis_title=dict(text='Y', font=dict(size=14)),
            xaxis=dict(range=[-1, grid_size], gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.2),  # Faded grid lines
            yaxis=dict(range=[-1, grid_size], gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.2),  # Faded grid lines
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor='#222831',  # Dark background color
            plot_bgcolor='#222831',  # Dark background color
            font=dict(color='white')  # White text color
        )

        fig2.update_layout(
            title=dict(text='District Association Map', font=dict(size=18)),
            xaxis_title=dict(text='X', font=dict(size=14)),
            yaxis_title=dict(text='Y', font=dict(size=14)),
            xaxis=dict(range=[-1, grid_size], gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.2),  # Faded grid lines
            yaxis=dict(range=[-1, grid_size], gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.2),  # Faded grid lines
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor='#222831',  # Dark background color
            plot_bgcolor='#222831',  # Dark background color
            font=dict(color='white')  # White text color
        )

        # Calculate metrics
        eg_result = find_eg()  # Placeholder input
        step_five = step_five_finder(0.1, 1, districts, p_0, n, c, r)
        refined_step_five = refined_step_five_finder(0.1, 1, districts, p_0, n, c, r)
        winners = find_winners()  # Placeholder input

        # Create calculations table
        calculations_table = dbc.Table([
            html.Thead([
                html.Tr([html.Th('Metric'), html.Th('Value')])
            ]),
            html.Tbody([
                html.Tr([html.Td('Efficiency Gap'), html.Td(f'{eg_result["eg"]:.4f}')]),
                html.Tr([html.Td('Step Five (Left, Right)'), html.Td(f'{step_five[0]:.4f}, {step_five[1]:.4f}')]),
                html.Tr([html.Td('Refined Step Five (Left, Right)'), html.Td(f'{refined_step_five[0]:.4f}, {refined_step_five[1]:.4f}')]),
                html.Tr([html.Td('Winners (Party 1, Party 0)'), html.Td(f'{winners[0]}, {winners[1]}')]),
            ])
        ], bordered=True, hover=True, responsive=True, striped=True)

        return fig1, fig2, '', None, calculations_table

    except Exception as e:
        if 'missed_nodes' in str(e):
            missed_nodes = eval(str(e).split('=')[1].strip())
            warning = f"Warning: {len(missed_nodes)} nodes were missed in the grid setup. This may affect the accuracy of the visualization."
            
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=dict(text='Error in Visualization', font=dict(size=18)),
                annotations=[dict(
                    text=warning,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )],
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            
            empty_fig.add_trace(go.Scatter(
                x=[node[0] for node in missed_nodes],
                y=[node[1] for node in missed_nodes],
                mode='markers',
                marker=dict(
                    color='yellow',
                    size=12,
                    line=dict(
                        color='black',
                        width=2
                    )
                ),
                name='Missed Nodes'
            ))
            
            return empty_fig, empty_fig, warning, missed_nodes, html.Div()
        else:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=dict(text='Error in Visualization', font=dict(size=18)),
                annotations=[dict(
                    text="The combination you selected is not valid. Please try again.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=20, color='red')  # Larger bright red text
                )],
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
                paper_bgcolor='#2c3e50',  # Dark background color
                plot_bgcolor='#2c3e50',  # Dark background color
            )
            return empty_fig, empty_fig, '', None, html.Div()
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run_server(debug=False, host='127.0.0.1', port=port)