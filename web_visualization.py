import dash
import dash_core_components as dcc
import dash_html_components as html

from find_sky_lvl import get_plt_image, get_filtered_gal_data

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cool galaxy!"),
    html.Div([
        "Input: ",
        html.Br(),
        html.Div([html.Div("v: ", style={'display': 'inline-block'}),
                  dcc.Input(id='v', value='(22, 16.5)', type="text")]),
        html.Div([html.Div("limits: ", style={'display': 'inline-block'}),
                  dcc.Input(id='limits', value='(16.5, 22)', type="text")]),
        html.Div([html.Div("count: ", style={'display': 'inline-block'}),
                  dcc.Input(id='count', value=15, type="number")])
    ]),
    html.Br(),
    html.Img(id='example')
])


@app.callback(
    dash.dependencies.Output('example', 'src'),
    [dash.dependencies.Input('v', 'value'),
     dash.dependencies.Input('limits', 'value'),
     dash.dependencies.Input('count', 'value')]
)
def update_figure(v, limits, count):
    v_filler = eval(v)
    limits_filler = eval(limits)
    count = int(count)
    gal_data_filter = get_filtered_gal_data()
    data = get_plt_image(gal_data_filter, v=v_filler, limits=limits_filler, count=count)
    return "data:image/png;base64,{}".format(data)


if __name__ == '__main__':
    app.run_server(debug=True)
