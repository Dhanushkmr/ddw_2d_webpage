import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import multiple_linear_regression as mlr


colors = {
    'background': '#FFFFFF',
    'text': '#7FDBFF'
}

df_1 = pd.read_csv("datasets/processed_data_MLR_2.csv")
df_ploting_1 = df_1.groupby('iso_code', as_index=False)['iso_code', 'total_deaths'].tail(1).reset_index()
world_map_1 = go.Figure(
    data=go.Choropleth(
        locations = df_ploting_1['iso_code'],
        z = df_ploting_1['total_deaths'],
        colorscale = 'Reds',
        hovertemplate = "%{location} - %{z}",
        autocolorscale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Total Number of deaths by country'
    )
)

df_conti = pd.read_csv("datasets/no-time-series.csv")

world_map_1.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})

markdown_text = '''
    ###### By *Lim Pin, Dhanush Kumar, Lester Leong, Tan Shu Yi, Yi Xiang*
'''

task_1_layout = html.Div(
    style={'backgroundColor': colors['background']},
    children = [
        html.Br(),
        dcc.Link('Go to Task 2', href='/apps/task_2'),
        dbc.Row(dbc.Col(html.H1(children="COVID-19 Dashboard [Task 1]"), width='auto'), justify='center'),
        dbc.Row(dbc.Col(dcc.Markdown(children=markdown_text), width='auto'), justify='center'),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Covid-19 Total Number of deaths by country"),
                        dcc.Graph(id="plotly-map", figure=world_map_1),
                    ]
                ), width = 10, align='center'
            ), align = 'center', justify='center'
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            html.Div([
                                "Select Continent:",
                                dcc.Dropdown(
                                    id = 'continent-dropdown', 
                                    options = [
                                        {'label': "Americas", 'value': "America"},
                                        {'label': "Europe", 'value': "Europe"},
                                        {'label': "Asia & Oceania", 'value': "Asia & Oceania"},
                                        {'label': "Africa", 'value': "Africa"},
                                    ], 
                                    value = "America"
                                ),
                            ]),
                            html.Div([
                                "Select a variable to see the relationship: ",
                                dcc.Dropdown(
                                    id = 'features-dropdown', 
                                    options = [
                                        {'label': "New Cases", 'value': "new_cases"},
                                        {'label': "Human Development Index", 'value': "human_development_index"},
                                        {'label': "Hospital Beds per thousand", 'value': "hospital_beds_per_thousand"},
                                        {'label': "Stringency Index", 'value': "stringency_index"},
                                    ], 
                                    value = "new_cases"
                                ),
                            ]),
                            html.Br(),
                            dbc.CardHeader(id = "deaths-for-continent"),
                            dcc.Graph(id="deaths_scatter"),
                        ]
                    ), width = 7, align='center'
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.H6("Adjust the sliders to see the predicted deaths... "),
                            html.Br(),
                            html.Br(),
                            html.Div(id='new-cases-slider-output-container'),
                            dcc.Slider(
                                children="New Cases:",
                                id='new-cases-slider',
                                min=0,
                                max=200000,
                                step=100,
                                value=10000,
                            ),
                            html.Div(id='hdi-slider-output-container'),
                            dcc.Slider(
                                children="Human Development Index",
                                id='hdi-slider',
                                min=0,
                                max=1,
                                step=0.01,
                                value=.85,
                            ),
                            html.Div(id='hosp-beds-slider-output-container'),
                            dcc.Slider(
                                id='hosp-beds-slider',
                                min=0,
                                max=10,
                                step=0.5,
                                value=2,
                            ),
                            html.Div(id='stringency-slider-output-container'),
                            dcc.Slider(
                                id='stringency-slider',
                                min=0,
                                max=100,
                                step=1,
                                value=50,
                            ),
                            
                            # dbc.Button("Generate new predictions", color="info", className="me-1", id = "generate-pred-button", n_clicks = 0),
                            html.Div(id='prediction-output-container')
                            
                        ]
                    ), width = 3, align='center'
                )
            ], align = 'center', justify='center' 
        )
    ]
)