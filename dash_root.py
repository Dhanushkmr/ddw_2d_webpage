import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


dash_root_layout = html.Div(
    [
        html.H3('hello world!'),
        dcc.Link('Go to Task 1', href='/task_1'),
        dcc.Link('Go to Task 2', href='/task_2')
    ]
)