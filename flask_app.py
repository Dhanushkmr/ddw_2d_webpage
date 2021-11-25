from dash import Dash, dcc, html
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import flask
from werkzeug.serving import run_simple
from task_1 import *
from task_2 import *

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


server = flask.Flask(__name__)

# dash_root = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/hello_world/')
dash_app1 = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/task_1/')
dash_app2 = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/task_2/')

# dash_root.layout = dash_root_layout
dash_app1.layout = task_1_layout
dash_app2.layout = task_2_layout


@dash_app2.callback(
    Output("new_deaths_scatter", "figure"),
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

    country_code = 'SGP'
    if clickData:
        country_code = clickData['points'][0]['location']
    new_deaths_scatter_fig = px.scatter(
        df[df.iso_code == country_code], x = df[df.iso_code == country_code].index, y = "new_deaths"
    )

    if button_id == "generate-pred-button":
        feature_columns = ["days", "stringency_index","percentage_vaccinated","positive_rate"]
        to_transform_features = ["days", "stringency_index","percentage_vaccinated","positive_rate"]
        pred_column = ["new_deaths"]
        df_country = df[df.iso_code == country_code]
        degree = 3 
        if n_clicks:
            tail, mu, sigma, beta_final = train_model(df_country,feature_columns, to_transform_features, pred_column, degree)
            new_get_pred = n_new_predictions(int(days), tail, degree, mu, sigma, feature_columns)
            pred_new_n = predict_norm(prepare_feature(new_get_pred), beta_final)
            data_new_pred = add_days(pred_new_n,int(days))
            df_country['new'] = 'Observed'
            o_fig = px.scatter(df_country, y ='new_deaths', x='days', color = 'new', color_discrete_map={'Observed':'blue'})
            n_fig = px.scatter(data_new_pred, y = 'new_deaths', x='days', color = 'new', color_discrete_map={'Predicted':'red'})
            combined_fig = go.Figure(data=o_fig.data + n_fig.data)
            combined_fig.update_layout(yaxis_title = "new_deaths",xaxis_title ="days")
            return combined_fig
        return None
    return new_deaths_scatter_fig


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