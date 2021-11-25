import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
from poly_reg_ridge_model import *

colors = {
    'background': '#FFFFFF',
    'text': '#7FDBFF'
}

df = pd.read_csv("latest_processed_data_with_weeks.csv")
df_ploting = df.groupby('iso_code')['total_deaths'].max().reset_index()
world_map = go.Figure(
    data=go.Choropleth(
        locations = df_ploting['iso_code'],
        z = df_ploting['total_deaths'],
        colorscale = 'Reds',
        hoverinfo = "location+z",
        autocolorscale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Total Number of Deaths'
    )
)
world_map.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})

markdown_text = '''
    ###### By *Lim Pin, Dhanush Kumar, Lester Leong, Tan Shuyi, Yi Xiang*
'''

task_2_layout = html.Div(
    style={'backgroundColor': colors['background']},
    children = [
        html.Br(),
        dcc.Link('Go to Task 1', href='/apps/task_1'),
        dbc.Row(dbc.Col(html.H1(children="COVID-19 Dashboard [Task 2]"), width='auto'), justify='center'),
        dbc.Row(dbc.Col(dcc.Markdown(children=markdown_text), width='auto'), justify='center'),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Covid-19 Total Deaths by country"),
                        dcc.Graph(id="plotly-map", figure=world_map),
                    ]
                ), width = 10, align='center'
            ), align = 'center', justify='center'
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(id = "new-deaths-for-country"),
                            dcc.Graph(id="new_deaths_scatter"),
                        ]
                    ), width = 8, align='center'
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.H6("Select number of days in the future to predict for: "),
                            dcc.Dropdown(id = 'days-dropdown', options = [{'label': str(i), 'value': i} for i in range(1, 20)], value = 10),
                            html.Br(),
                            dbc.Button("Generate new predictions", color="info", className="me-1", id = "generate-pred-button", n_clicks = 0),
                            
                        ]
                    ), width = 2, align='center'
                )
            ], align = 'center', justify='center' 
        )
    ]
)



