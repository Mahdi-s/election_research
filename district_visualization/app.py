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
from ortools.graph import pywrapgraph
from utility import setup, optimize, grid_setup, add_party_preference, create_district_map



app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  



# Define styles
SLIDER_BOX_STYLE = {
    'border': '2px solid #007BFF',
    'border-radius': '10px',
    'padding': '20px',
    'margin-bottom': '20px',
    'background-color': '#f8f9fa'
}

LABEL_STYLE = {
    'font-weight': 'bold',
    'margin-bottom': '5px'
}

GRAPH_STYLE = {
    'border': '1px solid #ddd',
    'border-radius': '5px',
    'padding': '10px',
    'margin-bottom': '20px'
}

def create_slider_with_tooltip(id, label, min_value, max_value, step, value, tooltip):
    if isinstance(min_value, int) and isinstance(max_value, int):
        marks = {i: str(i) for i in range(min_value, max_value+1, max(1, (max_value-min_value)//5))}
    else:
        num_marks = 5
        marks = {round(min_value + i * (max_value - min_value) / (num_marks - 1), 2): str(round(min_value + i * (max_value - min_value) / (num_marks - 1), 2)) for i in range(num_marks)}
    
    return html.Div([
        html.Div([
            html.Label(label, style=LABEL_STYLE),
            html.Span("?", id=f"{id}-tooltip", style={'cursor': 'pointer', 'margin-left': '5px'}),
        ]),
        dcc.Slider(
            id=id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks=marks
        ),
        dbc.Tooltip(tooltip, target=f"{id}-tooltip"),
    ], style={'margin-bottom': '10px'})

app.layout = html.Div([
    html.H1("District Visualization", style={'text-align': 'center', 'font-weight': 'bold'}),
    
    html.Div([
        create_slider_with_tooltip('grid-size-slider', "Grid Size", 5, 20, 1, 10, "The size of the grid for district visualization"),
        create_slider_with_tooltip('districts-slider', "Districts", 2, 10, 1, 5, "The number of districts to create"),
        create_slider_with_tooltip('p0-slider', "P0", 0.1, 0.9, 0.1, 0.5, "Probability parameter for district generation"),
        create_slider_with_tooltip('c-slider', "C", 1, 10, 1, 3, "Parameter C for district generation"),
        create_slider_with_tooltip('r-slider', "R", 1, 10, 1, 3, "Parameter R for district generation"),
        create_slider_with_tooltip('n-slider', "N", 1, 10, 1, 5, "Parameter N for district generation"),
    ], style=SLIDER_BOX_STYLE),
    
    html.Div(id='warning-message', style={'color': 'red', 'margin-top': '10px', 'font-weight': 'bold'}),
    
    html.Div([
        dcc.Graph(id='district-map', style=GRAPH_STYLE),
    ]),
    
    html.Hr(style={'border-top': '2px solid #007BFF', 'margin': '30px 0'}),
    
    html.Div([
        dcc.Graph(id='district-association-map', style=GRAPH_STYLE),
    ]),
    
    html.Div(id='calculations-table'),
    
    dcc.Store(id='missed-nodes-store')
])

def find_eg(win_count):
    # Implementation of find_eg function (placeholder)
    return {'eg': random.uniform(-0.1, 0.1)}

def step_five_finder(delta, gamma, dists_num, p_0, n, c, r):
    # Implementation of step_five_finder function
    proportion_a = 5
    proportion_b = (c*r) - proportion_a
    left_side = proportion_b / proportion_a
    F = math.sqrt((1-delta)/(2*dists_num))
    numerator = (F**2)-(4*3.14*gamma**(-1)*F*math.sqrt(2)*(1/n))-(8*((3.14)**2)*(gamma**(-1))*(1/n)**2)
    denominator = (F**2)+(4*3.14*gamma**(-1)*F*math.sqrt(2)*(1/n))+(8*((3.14)**2)*(gamma**(-1))*(1/n)**2)
    right_side = numerator/denominator
    return left_side, right_side

def refined_step_five_finder(delta, gamma, dists_num, p_0, n, c, r):
    # Implementation of refined_step_five_finder function
    proportion_a = 5
    proportion_b = (c*r) - proportion_a
    left_side = proportion_b / proportion_a
    F = math.sqrt((1-delta)/(2*dists_num))
    numerator = 2*(1/n)*(math.pi)
    denominator = (4*gamma)*(math.sqrt((1-delta)/(dists_num+1)))*n
    right_side = numerator/denominator
    return left_side, right_side

def find_winners(win_count):
    # Implementation of find_winners function (placeholder)
    return random.randint(0, 10), random.randint(0, 10)

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
            title=dict(text=f'District Map (Blue: {party_counts["blue"]}, Red: {party_counts["red"]})', font=dict(size=18, weight='bold')),
            xaxis_title=dict(text='X', font=dict(size=14, weight='bold')),
            yaxis_title=dict(text='Y', font=dict(size=14, weight='bold')),
            xaxis=dict(range=[-1, grid_size]), 
            yaxis=dict(range=[-1, grid_size])
        )

        fig2.update_layout(
            title=dict(text='District Association Map', font=dict(size=18, weight='bold')),
            xaxis_title=dict(text='X', font=dict(size=14, weight='bold')),
            yaxis_title=dict(text='Y', font=dict(size=14, weight='bold')),
            xaxis=dict(range=[-1, grid_size]), 
            yaxis=dict(range=[-1, grid_size])
        )

        # Calculate metrics
        eg_result = find_eg({'district1': {0.0: 100, 1.0: 80}})  # Placeholder input
        step_five = step_five_finder(0.1, 1, districts, p_0, n, c, r)
        refined_step_five = refined_step_five_finder(0.1, 1, districts, p_0, n, c, r)
        winners = find_winners({'iteration1': [{'district1': {0.0: 100, 1.0: 80}}]})  # Placeholder input

        # Create calculations table
        calculations_table = html.Table([
            html.Tr([html.Th('Metric'), html.Th('Value')]),
            html.Tr([html.Td('Efficiency Gap'), html.Td(f'{eg_result["eg"]:.4f}')]),
            html.Tr([html.Td('Step Five (Left, Right)'), html.Td(f'{step_five[0]:.4f}, {step_five[1]:.4f}')]),
            html.Tr([html.Td('Refined Step Five (Left, Right)'), html.Td(f'{refined_step_five[0]:.4f}, {refined_step_five[1]:.4f}')]),
            html.Tr([html.Td('Winners (Party 1, Party 0)'), html.Td(f'{winners[0]}, {winners[1]}')]),
        ], style={'border-collapse': 'collapse', 'width': '100%', 'margin-top': '20px'})

        return fig1, fig2, '', None, calculations_table

    except Exception as e:
        if 'missed_nodes' in str(e):
            missed_nodes = eval(str(e).split('=')[1].strip())
            warning = f"Warning: {len(missed_nodes)} nodes were missed in the grid setup. This may affect the accuracy of the visualization."
            
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=dict(text='Error in Visualization', font=dict(size=18, weight='bold')),
                annotations=[dict(
                    text=warning,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
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
                title=dict(text='Error in Visualization', font=dict(size=18, weight='bold')),
                annotations=[dict(
                    text=f"An error occurred: {str(e)}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
            return empty_fig, empty_fig, f"An error occurred: {str(e)}", None, html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)