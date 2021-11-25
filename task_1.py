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

task_1_layout = html.Div(
    html.H3('App 1')
)