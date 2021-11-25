from dash import Dash, dcc, html
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from werkzeug.serving import run_simple
from task_1 import *
from task_2 import *


server = flask.Flask(__name__)

# dash_root = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/hello_world/')
dash_app1 = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/task_1/')
dash_app2 = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/task_2/')

# dash_root.layout = dash_root_layout
dash_app1.layout = task_1_layout
dash_app2.layout = task_2_layout


@dash_app2.callback(
    Output("icu_scatter", "figure"),
    [
        Input("plotly-map", 'clickData'),
        Input("generate-pred-button", 'n_clicks'),
        Input("days-dropdown", "value")
    ]
)
def generate_new_pred(clickData, n_clicks, days):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs
    }, indent=2)
    print(ctx_msg)

    country_code = 'USA'
    if clickData:
        country_code = clickData['points'][0]['location']
    icu_scatter_fig = px.scatter(
        df_2[df_2.iso_code == country_code], x = df_2[df_2.iso_code == country_code].index, y = "icu_patients"
    )
    icu_scatter_fig.update_layout(title = country_code, yaxis_title = "ICU patients", xaxis_title ="days")

    if button_id == "generate-pred-button":
        feature_columns = ["days", "new_cases","percentage_population","hosp_patients"]
        to_transform_features = ["days", "new_cases","percentage_population","hosp_patients"]
        pred_column = ["icu_patients"]
        df_country = df_2[df_2.iso_code == country_code]
        degree = 3 
        if n_clicks:
            tail, mu, sigma, beta_final = train_model(df_country,feature_columns, to_transform_features, pred_column, degree)
            new_get_pred = n_new_predictions(int(days), tail, degree, mu, sigma, feature_columns)
            pred_new_n = predict_norm(prepare_feature(new_get_pred), beta_final)
            data_new_pred = add_days(pred_new_n,int(days))
            df_country['new'] = 'Observed'
            o_fig = px.scatter(df_country, y ='icu_patients', x='days', color = 'new', color_discrete_map={'Observed':'blue'})
            n_fig = px.scatter(data_new_pred, y = 'icu_patients', x='days', color = 'new', color_discrete_map={'Predicted':'red'})
            combined_fig = go.Figure(data=o_fig.data + n_fig.data,)
            combined_fig.update_layout(title = country_code, yaxis_title = "ICU patients",xaxis_title ="days")
            return combined_fig
        return None
    return icu_scatter_fig


@dash_app1.callback(
    Output("deaths_scatter", "figure"),
    [
        Input("features-dropdown", 'value'),
        Input("continent-dropdown", 'value')
    ]
)
def continent_deaths_viewer(features, continent):
    fig = px.scatter(df_conti[df_conti.continent_rearranged == continent], x = features, y = 'new_deaths')
    fig.update_layout(title = continent, yaxis_title = "New Deaths", xaxis_title = features)
    return fig


@dash_app1.callback(
    [
        Output("new-cases-slider-output-container", "children"),
        Output("hdi-slider-output-container", "children"),
        Output("hosp-beds-slider-output-container", "children"),
        Output("stringency-slider-output-container", "children"),
    ],
    [
        Input("new-cases-slider", 'value'),
        Input("hdi-slider", 'value'),
        Input("hosp-beds-slider", 'value'),
        Input("stringency-slider", 'value')
    ]
)
def slider_manager(new_cases, hdi, hospital, stringency):
    new_case_str = f"Number of new cases:{new_cases}"
    hdi_str = f"Human Development Index:{hdi}"
    hospital_str = f"Number of Hospital Beds (per thousand):{hospital}" 
    stringency_str = f"Stringency Index:{stringency}"
    return new_case_str, hdi_str, hospital_str, stringency_str


@dash_app1.callback(
    Output("prediction-output-container", "children"),
    [
        Input("new-cases-slider", 'value'),
        Input("hdi-slider", 'value'),
        Input("hosp-beds-slider", 'value'),
        Input("stringency-slider", 'value')
    ]
)
def slider_manager(new_cases, hdi, hospital, stringency):
    return '0'




@server.route('/')
@server.route('/hello_world/')
def hello():
    return '<h1><a href=/task_1/>Task 1</a></h1>\n<h1><a href=/task_2/>Task 2</a></h1>'

@server.route('/task_1/')
def render_dashboard_1():
    return flask.redirect('/task1')


@server.route('/task_2/')
def render_reports():
    return flask.redirect('/dash2')

app = DispatcherMiddleware(server, {
    # '/hello': dash_root.server,
    '/dash1': dash_app1.server,
    '/dash2': dash_app2.server
})

run_simple('0.0.0.0', 8080, app, use_reloader=True, use_debugger=True)