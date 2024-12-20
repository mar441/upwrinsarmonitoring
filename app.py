import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
from scipy.stats import t
import numpy as np
import os
from geopy.distance import geodesic

def load_displacement_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df = df.melt(id_vars=['Date'], 
                 var_name='pid', 
                 value_name='displacement')
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.drop(columns=['Date'], inplace=True)
    df['file'] = file_label
    df['pid'] = df['pid'].astype(str)
    return df

def load_anomaly_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df['file'] = file_label
    df['pid'] = df['pid'].astype(str)
    return df

geo_data_1 = pd.read_csv('mos_1.csv')
geo_data_2 = pd.read_csv('mos_2.csv')
geo_data_3 = pd.read_csv('msz_1.csv')
geo_data_4 = pd.read_csv('msz_2.csv')
geo_data_wroclaw = pd.concat([geo_data_1, geo_data_2, geo_data_3, geo_data_4], ignore_index=True)
geo_data_wroclaw['pid'] = geo_data_wroclaw['pid'].astype(str).str.strip()  

geo_data_turow = pd.read_csv('tr_73_1.csv')
geo_data_turow['pid'] = geo_data_turow['pid'].astype(str).str.strip() 

geo_data_turow_lstm = pd.read_csv('tr_geo_lstm.csv')
geo_data_turow_lstm['pid'] = geo_data_turow_lstm['pid'].astype(str).str.strip()

geo_data_bedzin = pd.read_csv('bedzin_geo.csv', delimiter=';')
geo_data_bedzin['pid'] = geo_data_bedzin['pid'].astype(str).str.strip()

geo_data_grunwald = pd.read_csv('grunwald_geo.csv', delimiter=';')
geo_data_grunwald['pid'] = geo_data_grunwald['pid'].astype(str).str.strip()

displacement_data_1 = load_displacement_data('mz2_10.csv', 'Descending 124')
displacement_data_2 = load_displacement_data('mz4_3.csv', 'Ascending 175')
displacement_data_3 = load_displacement_data('msz4_3.csv', 'Descending 124')
displacement_data_4 = load_displacement_data('msz2_3.csv', 'Ascending 175')

displacement_data_1['pid'] = displacement_data_1['pid'].astype(str).str.strip()
displacement_data_2['pid'] = displacement_data_2['pid'].astype(str).str.strip()
displacement_data_3['pid'] = displacement_data_3['pid'].astype(str).str.strip()
displacement_data_4['pid'] = displacement_data_4['pid'].astype(str).str.strip()

all_data_1 = pd.merge(displacement_data_1, geo_data_wroclaw, on='pid', how='left')
all_data_2 = pd.merge(displacement_data_2, geo_data_wroclaw, on='pid', how='left')
all_data_3 = pd.merge(displacement_data_3, geo_data_wroclaw, on='pid', how='left')
all_data_4 = pd.merge(displacement_data_4, geo_data_wroclaw, on='pid', how='left')

all_data_wroclaw = pd.concat([all_data_1, all_data_2, all_data_3, all_data_4], ignore_index=True)

displacement_data_5 = load_displacement_data('tr_73.csv', 'Ascending 73')
displacement_data_5['pid'] = displacement_data_5['pid'].astype(str).str.strip() 
all_data_turow = pd.merge(displacement_data_5, geo_data_turow, on='pid', how='left')

displacement_data_turow_lstm = load_displacement_data('tr_73_lstm.csv', 'Ascending 73 LSTM')
displacement_data_turow_lstm['pid'] = displacement_data_turow_lstm['pid'].astype(str).str.strip() 
all_data_turow_lstm = pd.merge(displacement_data_turow_lstm, geo_data_turow_lstm, on='pid', how='left')

displacement_data_bedzin = load_displacement_data('bedzin_displ.csv', 'Ascending 175')
displacement_data_bedzin['pid'] = displacement_data_bedzin['pid'].astype(str).str.strip() 
all_data_bedzin = pd.merge(displacement_data_bedzin, geo_data_bedzin, on='pid', how='left')

displacement_data_grunwald = load_displacement_data('grunwald_displ.csv', 'Ascending 175')
displacement_data_grunwald['pid'] = displacement_data_grunwald['pid'].astype(str).str.strip() 
all_data_grunwald = pd.merge(displacement_data_grunwald, geo_data_grunwald, on='pid', how='left')

prediction_data_1 = pd.read_csv('predictions_values.csv')
prediction_data_1 = prediction_data_1.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_1['label'] = 'Prediction Set 1'
prediction_data_1['step'] = prediction_data_1.groupby('pid').cumcount()

prediction_data_2 = pd.read_csv('predictions_values2.csv') 
prediction_data_2 = prediction_data_2.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_2['label'] = 'Prediction Set 2'
prediction_data_2['step'] = prediction_data_2.groupby('pid').cumcount()

prediction_data_3 = pd.read_csv('predictions_values3.csv') 
prediction_data_3 = prediction_data_3.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_3['label'] = 'Prediction Set 3'
prediction_data_3['step'] = prediction_data_3.groupby('pid').cumcount()

prediction_data_4 = pd.read_csv('predictions_values4.csv') 
prediction_data_4 = prediction_data_4.melt(var_name='pid', 
                                           value_name='predicted_displacement')
prediction_data_4['label'] = 'Prediction Set 4'
prediction_data_4['step'] = prediction_data_4.groupby('pid').cumcount()

all_prediction_data_wroclaw = pd.concat([prediction_data_1, prediction_data_2, prediction_data_3, prediction_data_4], ignore_index=True)

prediction_data_turow = pd.read_csv('predictions_values5.csv', delimiter=';') 
prediction_data_turow = prediction_data_turow.melt(var_name='pid',
                                                   value_name='predicted_displacement')
prediction_data_turow['label'] = 'Prediction Set 5'
prediction_data_turow['step'] = prediction_data_turow.groupby('pid').cumcount()

prediction_data_turow_lstm = pd.read_csv('predykcja_lstm.csv', delimiter=';')
prediction_data_turow_lstm = prediction_data_turow_lstm.melt(var_name='pid', value_name='predicted_displacement')
prediction_data_turow_lstm['label'] = 'LSTM Prediction Set'
prediction_data_turow_lstm['step'] = prediction_data_turow_lstm.groupby('pid').cumcount()

prediction_data_bedzin = pd.read_csv('predictions_bedzin.csv', delimiter=',')
prediction_data_bedzin = prediction_data_bedzin.melt(var_name='pid', value_name='predicted_displacement')
prediction_data_bedzin['label'] = 'Bedzin Prediction Set'
prediction_data_bedzin['step'] = prediction_data_bedzin.groupby('pid').cumcount()

prediction_data_grunwald = pd.read_csv('predictions_grunwald.csv', delimiter=',')
prediction_data_grunwald = prediction_data_grunwald.melt(var_name='pid', value_name='predicted_displacement')
prediction_data_grunwald['label'] = 'Grunwald Prediction Set'
prediction_data_grunwald['step'] = prediction_data_grunwald.groupby('pid').cumcount()

anomaly_data_1_95 = load_anomaly_data('anomaly_output_95.csv', 'Anomaly Set 1 (95%)')
anomaly_data_2_95 = load_anomaly_data('anomaly_output2_95.csv', 'Anomaly Set 2 (95%)')
anomaly_data_3_95 = load_anomaly_data('anomaly_output3_95.csv', 'Anomaly Set 3 (95%)')
anomaly_data_4_95 = load_anomaly_data('anomaly_output4_95.csv', 'Anomaly Set 4 (95%)')

all_anomaly_data_95_wroclaw = pd.concat([anomaly_data_1_95, anomaly_data_2_95, anomaly_data_3_95, anomaly_data_4_95], ignore_index=True)

anomaly_data_turow_95  = load_anomaly_data('anomaly_output5_95.csv', 'Anomaly Set 5 (95%)')
anomaly_data_turow_95 = anomaly_data_turow_95.groupby('pid').head(31)

anomaly_data_1_99 = load_anomaly_data('anomaly_output_99.csv', 'Anomaly Set 1 (99%)')
anomaly_data_2_99 = load_anomaly_data('anomaly_output2_99 .csv', 'Anomaly Set 2 (99%)')
anomaly_data_3_99 = load_anomaly_data('anomaly_output3_99.csv', 'Anomaly Set 3 (99%)')
anomaly_data_4_99 = load_anomaly_data('anomaly_output4_99.csv', 'Anomaly Set 4 (99%)')

all_anomaly_data_99_wroclaw = pd.concat([anomaly_data_1_99, anomaly_data_2_99, anomaly_data_3_99, anomaly_data_4_99], ignore_index=True)

anomaly_data_turow_99 = load_anomaly_data('anomaly_output5_99.csv', 'Anomaly Set 5 (99%)')
anomaly_data_turow_99 = anomaly_data_turow_99.groupby('pid').head(31)

anomaly_data_turow_95_lstm = load_anomaly_data('anomaly_lstm_95.csv', 'Anomaly Set 5 LSTM (95%)')
anomaly_data_turow_95_lstm = anomaly_data_turow_95_lstm.groupby('pid').head(31)

anomaly_data_turow_99_lstm = load_anomaly_data('anomaly_lstm_99.csv', 'Anomaly Set 5 LSTM (99%)')
anomaly_data_turow_99_lstm = anomaly_data_turow_99_lstm.groupby('pid').head(31)

anomaly_data_bedzin_95 = load_anomaly_data('anomaly_bedzin_95.csv', 'Anomaly Set 6 LSTM (95%)')
anomaly_data_bedzin_95 = anomaly_data_bedzin_95.groupby('pid').head(11)

anomaly_data_bedzin_99 = load_anomaly_data('anomaly_bedzin_99.csv', 'Anomaly Set 6 LSTM (99%)')
anomaly_data_bedzin_99 = anomaly_data_bedzin_99.groupby('pid').head(11)

anomaly_data_grunwald_95 = load_anomaly_data('anomaly_grunwald_95.csv', 'Anomaly Set 6 LSTM (95%)')
anomaly_data_grunwald_95 = anomaly_data_grunwald_95.groupby('pid').head(61)

anomaly_data_grunwald_99 = load_anomaly_data('anomaly_grunwald_99.csv', 'Anomaly Set 6 LSTM (99%)')
anomaly_data_grunwald_99 = anomaly_data_grunwald_99.groupby('pid').head(61)

all_data_wroclaw.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_wroclaw['displacement_diff'] = all_data_wroclaw.groupby('pid')['displacement'].diff().round(1)
all_data_wroclaw['time_diff'] = all_data_wroclaw.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_wroclaw['displacement_speed'] = ((all_data_wroclaw['displacement_diff'] / all_data_wroclaw['time_diff']) * 365).round(1)

mean_velocity_data_wroclaw = all_data_wroclaw.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_wroclaw.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_wroclaw = pd.merge(all_data_wroclaw, mean_velocity_data_wroclaw, on='pid', how='left')

all_data_turow.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_turow['displacement_diff'] = all_data_turow.groupby('pid')['displacement'].diff().round(1)
all_data_turow['time_diff'] = all_data_turow.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_turow['displacement_speed'] = ((all_data_turow['displacement_diff'] / all_data_turow['time_diff']) * 365).round(1)

mean_velocity_data_turow = all_data_turow.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_turow.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_turow = pd.merge(all_data_turow, mean_velocity_data_turow, on='pid', how='left')

all_data_turow_lstm.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_turow_lstm['displacement_diff'] = all_data_turow_lstm.groupby('pid')['displacement'].diff().round(1)
all_data_turow_lstm['time_diff'] = all_data_turow_lstm.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_turow_lstm['displacement_speed'] = ((all_data_turow_lstm['displacement_diff'] / all_data_turow_lstm['time_diff']) * 365).round(1)

mean_velocity_data_turow_lstm = all_data_turow_lstm.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_turow_lstm.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_turow_lstm = pd.merge(all_data_turow_lstm, mean_velocity_data_turow_lstm, on='pid', how='left')

all_data_bedzin.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_bedzin['displacement_diff'] = all_data_bedzin.groupby('pid')['displacement'].diff().round(1)
all_data_bedzin['time_diff'] = all_data_bedzin.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_bedzin['displacement_speed'] = ((all_data_bedzin['displacement_diff'] / all_data_bedzin['time_diff']) * 365).round(1)

mean_velocity_data_bedzin = all_data_bedzin.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_bedzin.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_bedzin = pd.merge(all_data_bedzin, mean_velocity_data_bedzin, on='pid', how='left')

all_data_grunwald.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_grunwald['displacement_diff'] = all_data_grunwald.groupby('pid')['displacement'].diff().round(1)
all_data_grunwald['time_diff'] = all_data_grunwald.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_grunwald['displacement_speed'] = ((all_data_grunwald['displacement_diff'] / all_data_grunwald['time_diff']) * 365).round(1)

mean_velocity_data_grunwald = all_data_grunwald.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_grunwald.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_grunwald = pd.merge(all_data_grunwald, mean_velocity_data_grunwald, on='pid', how='left')

def compute_prefix_sums(data):
    data = data.sort_values(by=['pid', 'step'])
    pivot = data.pivot(index='pid', columns='step', values='predicted_displacement').fillna(0).round(1)
    pivot[0] = 0
    pivot = pivot.sort_index(axis=1)
    for col in pivot.columns[1:]:
        pivot[col] = (pivot[col] + pivot[col-1]).round(1)
    return pivot

wroclaw_prefix = compute_prefix_sums(all_prediction_data_wroclaw)
turow_prefix = compute_prefix_sums(prediction_data_turow)
turow_lstm_prefix = compute_prefix_sums(prediction_data_turow_lstm)
bedzin_prefix = compute_prefix_sums(prediction_data_bedzin)
grunwald_prefix = compute_prefix_sums(prediction_data_grunwald)

prefix_data = {
    ('wroclaw', 'autoencoder'): wroclaw_prefix,
    ('turow', 'autoencoder'): turow_prefix,
    ('turow', 'lstm'): turow_lstm_prefix,
    ('bedzin', 'autoencoder'): bedzin_prefix,
    ('grunwald', 'autoencoder'): grunwald_prefix,
}

MAX_WROCLAW = wroclaw_prefix.columns.max()
MAX_TUROW = turow_prefix.columns.max()
MAX_BEDZIN = bedzin_prefix.columns.max()
MAX_GRUNWALD = grunwald_prefix.columns.max()

def add_obs_step(df):
    df = df.sort_values(by=['pid', 'timestamp'])
    df['obs_step'] = df.groupby('pid').cumcount() + 1
    return df

all_data_wroclaw = add_obs_step(all_data_wroclaw)
all_data_turow = add_obs_step(all_data_turow)
all_data_bedzin = add_obs_step(all_data_bedzin)
all_data_grunwald = add_obs_step(all_data_grunwald)

def compute_prefix_sums_actual(df):
    pivot = df.pivot(index='pid', columns='obs_step', values='displacement').fillna(0).round(1)
    pivot[0] = 0
    pivot = pivot.sort_index(axis=1)
    for col in pivot.columns[1:]:
        pivot[col] = (pivot[col] + pivot[col-1]).round(1)
    return pivot

actual_wroclaw_prefix = compute_prefix_sums_actual(all_data_wroclaw)
actual_turow_prefix = compute_prefix_sums_actual(all_data_turow)
actual_bedzin_prefix = compute_prefix_sums_actual(all_data_bedzin)
actual_grunwald_prefix = compute_prefix_sums_actual(all_data_grunwald)

actual_prefix_data = {
    'wroclaw': actual_wroclaw_prefix,
    'turow': actual_turow_prefix,
    'bedzin': actual_bedzin_prefix,
    'grunwald': actual_grunwald_prefix,
}

MAX_ACTUAL_WROCLAW = actual_wroclaw_prefix.columns.max()
MAX_ACTUAL_TUROW = actual_turow_prefix.columns.max()
MAX_ACTUAL_BEDZIN = actual_bedzin_prefix.columns.max()
MAX_ACTUAL_GRUNWALD = actual_grunwald_prefix.columns.max()

px.set_mapbox_access_token('pk.eyJ1IjoibWFycGllayIsImEiOiJjbTBxbXBsMGQwYjgyMmxzN3RpdmlhZDVrIn0.YWJh1RM6HKfN_pbH-jtJ6A')

orbit_geometry_info = {
    'Ascending 175': {
        'Relative orbit number': '175',
        'View angle': '348.9°',
        'Mean Incidence angle': '33.18°'  
    },
    'Descending 124': {
        'Relative orbit number': '124',
        'View angle': '189.35°',  
        'Mean Incidence angle': '42.64°'  
    },
    'Ascending 73': {
        'Relative orbit number': '73',
        'View angle': '350.49°', 
        'Mean Incidence angle': '41.85°' 
    }}

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H3("Select Map and Data Visualization Options"),

    html.Div([
        html.Div([
            html.Label("Map Style"),
            dcc.Dropdown(
                id='map-style-dropdown',
                options=[
                    {'label': 'Satellite', 'value': 'satellite'},
                    {'label': 'Outdoors', 'value': 'outdoors'},
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Dark', 'value': 'dark'},
                    {'label': 'Streets', 'value': 'streets'}
                ],
                value='satellite',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Visualization Option"),
            dcc.Dropdown(
                id='color-mode-dropdown',
                options=[
                    {'label': 'Orbit Type', 'value': 'orbit'},
                    {'label': 'Displacement Mean Velocity [mm/year]', 'value': 'speed'},
                    {'label': 'Anomaly Type', 'value': 'anomaly_type'},
                    {'label': 'Prediction Velocity', 'value': 'prediction_velocity'},
                    {'label': 'Cumulative Displacement', 'value': 'actual_displacement_velocity'}
                ],
                value='orbit',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Filter by LOS Geometry"),
            dcc.Dropdown(
                id='orbit-filter-dropdown',
                options=[
                    {'label': 'Ascending 175', 'value': 'Ascending 175'},
                    {'label': 'Descending 124', 'value': 'Descending 124'}
                ],
                value='Ascending 175',
                multi=True,
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Select Area of Interest"),
            dcc.Dropdown(
                id='area-dropdown',
                options=[
                    {'label': 'Mosty - Wrocław', 'value': 'wroclaw'},
                    {'label': 'Turów', 'value': 'turow'},
                    {'label': 'Plac Grunwaldzki - Wrocław', 'value': 'grunwald'},
                    {'label': 'Będzin', 'value': 'bedzin'}
                ],
                value='wroclaw',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Enable Distance Calculation"),
            dcc.Dropdown(
                id='distance-calc-dropdown',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),

    html.Div(id='distance-output', style={'font-size': '16px', 'padding': '10px', 'color': 'black'}),

    html.Div(id='prediction-method-container', children=[
        html.Label("Select Prediction Method"),
        dcc.Dropdown(
            id='prediction-method-dropdown',
            options=[
                {'label': 'Autoencoder', 'value': 'autoencoder'},
                {'label': 'LSTM', 'value': 'lstm'}
            ],
            value='autoencoder',
            clearable=False,
            style={'width': '100%'}
        )
    ], style={'display': 'none', 'padding': '10px'}),

    html.Div([
        html.Label("Select Observation Range"),
        dcc.RangeSlider(
            id='dynamic-prediction-range-slider',
            min=1,
            max=60,
            step=1, 
            marks={}, 
            value=[1,5],
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False   
        )
    ], id='prediction-slider-container', style={'display': 'none', 'padding': '10px'}),  

    dcc.Graph(id='map', style={'height': '80vh', 'width': '95vw'}, config={'scrollZoom': True}),
    dcc.Store(id='selected-points', data={'point_1': None, 'point_2': None}),

    html.Div(id='displacement-container', children=[
        html.Div([ 
            html.Label("Select Date Range", style={'font-size': '16px'}),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=all_data_wroclaw['timestamp'].min(),
                end_date=all_data_wroclaw['timestamp'].max(),
                display_format='YYYY-MM-DD',
                style={'height': '5px', 'width': '300px', 'font-family': 'Arial', 'font-size': '4px', 'display': 'inline-block', 'padding': '5px'}
            )
        ], style={'display': 'inline-block', 'padding': '10px'}),

        html.Div([ 
            html.Label("Set Y-Axis Range (mm)"),
            dcc.Input(
                id='y-axis-min',
                type='number',
                placeholder='Min',
                style={'width': '20%', 'margin-right': '10px'}
            ),
            dcc.Input(
                id='y-axis-max',
                type='number',
                placeholder='Max',
                style={'width': '20%'}
            ),
        ], style={'display': 'inline-block', 'padding': '10px'}),

        dcc.Graph(id='displacement-graph', style={'height': '50vh', 'width': '95vw'})
    ], style={'display': 'none'})
])

@app.callback(
    Output('prediction-slider-container', 'style'),
    [Input('color-mode-dropdown', 'value')]
)
def toggle_prediction_slider_visibility(color_mode):
    if color_mode in ['prediction_velocity', 'actual_displacement_velocity']: 
        return {'display': 'block', 'padding': '10px'}
    else:
        return {'display': 'none', 'padding': '10px'}

@app.callback(
    Output('prediction-method-container', 'style'),
    Input('area-dropdown', 'value')
)
def toggle_prediction_method_dropdown(selected_area):
    if selected_area == 'turow':
        return {'display': 'block', 'padding': '10px'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('orbit-filter-dropdown', 'options'), 
     Output('orbit-filter-dropdown', 'value'),
     Output('orbit-filter-dropdown', 'disabled')],
    [Input('area-dropdown', 'value')]
)
def update_orbit_filter(selected_area):
    if selected_area == 'turow':
        return [{'label': 'Ascending 73', 'value': 'Ascending 73'}], 'Ascending 73', True
    elif selected_area == 'bedzin':
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 175', True
    elif selected_area == 'grunwald':
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 175', True
    else:
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}, 
                {'label': 'Descending 124', 'value': 'Descending 124'}], 'Ascending 175', False

@app.callback(
    [Output('dynamic-prediction-range-slider', 'max'),
     Output('dynamic-prediction-range-slider', 'marks'),
     Output('dynamic-prediction-range-slider', 'value')],
    [Input('area-dropdown', 'value'),
     Input('color-mode-dropdown', 'value'),
     Input('prediction-method-dropdown', 'value')]
)
def update_slider_max(selected_area, color_mode, prediction_method):

    if color_mode == 'actual_displacement_velocity':
        max_val = {
            'turow': MAX_ACTUAL_TUROW,
            'bedzin': MAX_ACTUAL_BEDZIN,
            'grunwald': MAX_ACTUAL_GRUNWALD
        }.get(selected_area, MAX_ACTUAL_WROCLAW)
    else:
        max_val = 60  
        
    data_for_area = {
        'wroclaw': all_data_wroclaw,
        'turow': all_data_turow,
        'bedzin': all_data_bedzin,
        'grunwald': all_data_grunwald
    }.get(selected_area, all_data_wroclaw)

    timestamps_df = data_for_area.drop_duplicates(subset='obs_step')[['obs_step', 'timestamp']].sort_values('obs_step')

    if max_val > 60:
        N = 40 
    elif max_val > 20:
        N = 15  
    else:
        N = 5  

    marks = {}
    for _, row in timestamps_df.iterrows():
        step_val = row['obs_step']
        if step_val <= max_val:
            if step_val == 1 or step_val == max_val or step_val % N == 0:
                marks[step_val] = row['timestamp'].strftime('%Y-%m-%d')  
            else:
                marks[step_val] = "" 

    default_end = min(5, max_val)
    return max_val, marks, [1, default_end]

@app.callback(
    Output('map', 'figure'),
    [
        Input('map-style-dropdown', 'value'),
        Input('color-mode-dropdown', 'value'),
        Input('orbit-filter-dropdown', 'value'),
        Input('area-dropdown', 'value'),
        Input('dynamic-prediction-range-slider', 'value'),
        Input('prediction-method-dropdown', 'value')
    ]
)
def update_map(map_style, color_mode, orbit_filter, selected_area, pred_range, prediction_method):
    if selected_area == 'wroclaw':
        data = all_data_wroclaw.drop_duplicates(subset=['pid'])
        center_coords = {'lat': all_data_wroclaw['latitude'].mean(), 'lon': all_data_wroclaw['longitude'].mean()}
        zoom_level = 14
    elif selected_area == 'turow':
        data = all_data_turow.drop_duplicates(subset=['pid'])
        center_coords = {'lat': 50.90803234267631, 'lon': 14.898742567091745}
        zoom_level = 12
        orbit_filter = ['Ascending 73']
    elif selected_area == 'grunwald':
        data = all_data_grunwald.drop_duplicates(subset=['pid'])
        center_coords = {'lat': 51.11249671461431, 'lon': 17.06133312265709}
        zoom_level = 14
        orbit_filter = ['Ascending 175']
    else:
        data = all_data_bedzin.drop_duplicates(subset=['pid'])
        center_coords = {'lat': all_data_bedzin['latitude'].mean(), 'lon': all_data_bedzin['longitude'].mean()}
        zoom_level = 14.5
        orbit_filter = ['Ascending 175']

    if isinstance(orbit_filter, str):
        orbit_filter = [orbit_filter]

    filtered_data = data[data['file'].isin(orbit_filter)].copy()
    filtered_data.loc[:, 'mean_velocity'] = filtered_data['mean_velocity'].round(1)

    start_val, end_val = pred_range

    if color_mode == 'prediction_velocity':
        if selected_area == 'turow':
            max_steps = MAX_TUROW
        elif selected_area == 'bedzin':
            max_steps = MAX_BEDZIN
        elif selected_area == 'grunwald':
            max_steps = MAX_GRUNWALD
        else:
            max_steps = MAX_WROCLAW

        pred_key = (selected_area, prediction_method if selected_area == 'turow' else 'autoencoder')
        prefix_pivot = prefix_data[pred_key]

        end_val = min(end_val, max_steps)
        start_val = min(start_val, max_steps)

        numerator = prefix_pivot[end_val] - prefix_pivot[start_val-1]
        denominator = (end_val - start_val + 1)
        prediction_avg = numerator / denominator

        merged_data = filtered_data.set_index('pid')
        merged_data['prediction_velocity'] = prediction_avg
        merged_data.reset_index(inplace=True)

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='prediction_velocity',  
            color_continuous_scale='Jet',
            range_color=(-5, 5),
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Prediction Velocity Average')

    elif color_mode == 'actual_displacement_velocity':
        if selected_area == 'turow':
            max_steps = MAX_ACTUAL_TUROW
            prefix_pivot = actual_prefix_data['turow']
        elif selected_area == 'bedzin':
            max_steps = MAX_ACTUAL_BEDZIN
            prefix_pivot = actual_prefix_data['bedzin']
        elif selected_area == 'grunwald':
            max_steps = MAX_ACTUAL_GRUNWALD
            prefix_pivot = actual_prefix_data['grunwald']
        else:
            max_steps = MAX_ACTUAL_WROCLAW
            prefix_pivot = actual_prefix_data['wroclaw']

        end_val = min(end_val, max_steps)
        start_val = min(start_val, max_steps)

        numerator = prefix_pivot[end_val] - prefix_pivot[start_val-1]
        denominator = (end_val - start_val + 1)
        actual_avg = numerator / denominator

        merged_data = filtered_data.set_index('pid')
        merged_data['actual_displacement_velocity'] = actual_avg
        merged_data.reset_index(inplace=True)

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='actual_displacement_velocity',  
            color_continuous_scale='Jet',
            range_color=(-5, 5),
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Actual Displacement Velocity Average')

    elif color_mode == 'anomaly_type':
        if selected_area == 'wroclaw':
            merged_data = filtered_data.merge(all_anomaly_data_99_wroclaw[['pid', 'is_anomaly']], on='pid', how='left')
        elif selected_area == 'turow':
            merged_data = filtered_data.merge(anomaly_data_turow_99[['pid', 'is_anomaly']], on='pid', how='left')
        elif selected_area == 'grunwald':
            merged_data = filtered_data.merge(anomaly_data_grunwald_99[['pid', 'is_anomaly']], on='pid', how='left')
        else:
            merged_data = filtered_data.merge(anomaly_data_bedzin_99[['pid', 'is_anomaly']], on='pid', how='left')

        merged_data['is_anomaly'] = merged_data['is_anomaly'].fillna(False).astype(bool)
        merged_data['consecutive_anomalies'] = (
            merged_data.groupby('pid')['is_anomaly']
            .rolling(3, min_periods=3).sum().reset_index(0, drop=True)
        )
        merged_data['anomaly_3plus'] = merged_data['consecutive_anomalies'] >= 3

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            color=merged_data['anomaly_3plus'].map({True: 'Anomaly', False: 'No Anomaly'}),
            color_discrete_map={'Anomaly': 'red', 'No Anomaly': 'green'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Anomaly Type')

    elif color_mode == 'orbit':
        fig = px.scatter_mapbox(
            filtered_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            color='file',
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Orbit Type')

    elif color_mode == 'speed':
        fig = px.scatter_mapbox(
            filtered_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='mean_velocity',
            color_continuous_scale='Jet',
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Mean Velocity')
        
    if orbit_filter is not None:
        if isinstance(orbit_filter, str):
            orbit_filter = [orbit_filter]

        annotation_lines = ["Orbit Geometry Info:<br>"]
        for orbit in orbit_filter:
            if orbit in orbit_geometry_info:
                info = orbit_geometry_info[orbit]
                annotation_lines.append(
                    f"<b>{orbit}</b>:<br>"
                    f"Relative orbit number: {info['Relative orbit number']}<br>"
                    f"View angle: {info['View angle']}<br>"
                    f"Mean Incidence angle: {info['Mean Incidence angle']}<br><br>"
                )

        if len(annotation_lines) > 1:
            annotation_text = "".join(annotation_lines)
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=1, y=1,
                showarrow=False,
                align="left",
                bordercolor="#cccccc",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )

    fig.update_layout(
        mapbox_style=map_style,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(center=center_coords),
        coloraxis_colorbar=dict(title=None), 
    )

    return fig

@app.callback(
    Output('selected-points', 'data'),
    [Input('map', 'clickData')],
    [State('selected-points', 'data')]
)
def update_selected_points(clickData, selected_points):
    if clickData is None:
        return selected_points
    
    point_id = clickData['points'][0]['hovertext']
    lat = clickData['points'][0]['lat']
    lon = clickData['points'][0]['lon']
    
    if selected_points['point_1'] is None:
        selected_points['point_1'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    elif selected_points['point_2'] is None:
        selected_points['point_2'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    else:
        selected_points = {'point_1': None, 'point_2': None}

    return selected_points

@app.callback(
    Output('distance-output', 'children'),
    [Input('selected-points', 'data'),
     Input('distance-calc-dropdown', 'value')]
)
def display_distance(selected_points, distance_calc_enabled):
    if distance_calc_enabled == 'no':
        return ""

    point_1 = selected_points['point_1']
    point_2 = selected_points['point_2']
    
    if point_1 is not None and point_2 is not None:
        coords_1 = (point_1['lat'], point_1['lon'])
        coords_2 = (point_2['lat'], point_2['lon'])

        distance_km = geodesic(coords_1, coords_2).kilometers
        
        return html.Div([
            html.H4("Selected Points and Distance"),
            html.Ul([
                html.Li(f"Point 1: {point_1['pid']} (Lat: {point_1['lat']}, Lon: {point_1['lon']})"),
                html.Li(f"Point 2: {point_2['pid']} (Lat: {point_2['lat']}, Lon: {point_2['lon']})"),
                html.Li(f"Distance: {distance_km:.2f} km")
            ], style={'list-style-type': 'none', 'padding': '0', 'margin': '0'})
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'})
    else:
        return "Select two points on the map to calculate the distance."

@app.callback(
    [Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed')],
    [Input('area-dropdown', 'value')]
)
def update_date_picker(selected_area):
    if selected_area == 'wroclaw':
        start_date = all_data_wroclaw['timestamp'].min()
        end_date = all_data_wroclaw['timestamp'].max()
    elif selected_area == 'bedzin':
        start_date = all_data_bedzin['timestamp'].min()
        end_date = all_data_bedzin['timestamp'].max()
    elif selected_area == 'grunwald':
        start_date = all_data_grunwald['timestamp'].min()
        end_date = all_data_grunwald['timestamp'].max()
    else:
        start_date = all_data_turow['timestamp'].min()
        end_date = all_data_turow['timestamp'].max()

    return start_date, end_date, start_date, end_date

@app.callback(
    [Output('displacement-graph', 'figure'), Output('displacement-container', 'style')],
    [Input('map', 'clickData'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('y-axis-min', 'value'),
     Input('y-axis-max', 'value'),
     Input('area-dropdown', 'value'),
     Input('prediction-method-dropdown', 'value')] 
)
def display_displacement(clickData, start_date, end_date, y_min, y_max, selected_area, prediction_method):
    if clickData is None:
        return {}, {'display': 'none'}

    point_id = clickData['points'][0]['hovertext']
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if selected_area == 'wroclaw':
        full_data = all_data_wroclaw[all_data_wroclaw['pid'] == point_id].copy() 
        anomaly_data_95 = all_anomaly_data_95_wroclaw
        anomaly_data_99 = all_anomaly_data_99_wroclaw
        last_n_data = full_data.tail(60)
    elif selected_area == 'grunwald':
        full_data = all_data_grunwald[all_data_grunwald['pid'] == point_id].copy() 
        anomaly_data_95 = anomaly_data_grunwald_95
        anomaly_data_99 = anomaly_data_grunwald_99
        last_n_data = full_data.tail(61)
    elif selected_area == 'bedzin':
        full_data = all_data_bedzin[all_data_bedzin['pid'] == point_id].copy() 
        anomaly_data_95 = anomaly_data_bedzin_95
        anomaly_data_99 = anomaly_data_bedzin_99
        last_n_data = full_data.tail(11)
    else:
        if prediction_method == 'autoencoder':
            full_data = all_data_turow[all_data_turow['pid'] == point_id].copy()
            anomaly_data_95 = anomaly_data_turow_95
            anomaly_data_99 = anomaly_data_turow_99
            last_n_data = full_data.tail(31)
        elif prediction_method == 'lstm':
            full_data = all_data_turow_lstm[all_data_turow_lstm['pid'] == point_id].copy()
            anomaly_data_95 = anomaly_data_turow_95_lstm
            anomaly_data_99 = anomaly_data_turow_99_lstm
            last_n_data = full_data.tail(31)

    last_n_data.set_index('timestamp', inplace=True)
    filtered_data = full_data[(full_data['timestamp'] >= start_date) & (full_data['timestamp'] <= end_date)].copy()

    if not last_n_data.empty and (last_n_data.index.min() <= end_date) and (last_n_data.index.max() >= start_date):
        filtered_last_n_data = last_n_data[(last_n_data.index >= start_date) & (last_n_data.index <= end_date)].copy()
    else:
        filtered_last_n_data = pd.DataFrame()

    filtered_anomalies_95 = anomaly_data_95[anomaly_data_95['pid'] == point_id].copy()
    filtered_anomalies_99 = anomaly_data_99[anomaly_data_99['pid'] == point_id].copy()

    if not filtered_anomalies_95.empty:
        filtered_anomalies_95 = filtered_anomalies_95.tail(len(filtered_last_n_data)).copy()
        if len(filtered_anomalies_95) > 0:
            filtered_anomalies_95['timestamp'] = filtered_last_n_data.index.values[:len(filtered_anomalies_95)]
            filtered_anomalies_95.set_index('timestamp', inplace=True)
            filtered_last_n_data = filtered_last_n_data.join(
                filtered_anomalies_95[['predicted_value', 'upper_bound', 'lower_bound', 'is_anomaly']], 
                how='left'
            )

    if not filtered_anomalies_99.empty:
        filtered_anomalies_99 = filtered_anomalies_99.tail(len(filtered_last_n_data)).copy()
        if len(filtered_anomalies_99) > 0:
            filtered_anomalies_99['timestamp'] = filtered_last_n_data.index.values[:len(filtered_anomalies_99)]
            filtered_anomalies_99.set_index('timestamp', inplace=True)
            filtered_last_n_data = filtered_last_n_data.join(
                filtered_anomalies_99[['upper_bound', 'lower_bound', 'is_anomaly']], 
                how='left', rsuffix='_99'
            )

    fig = px.line(filtered_data, x='timestamp', y='displacement', 
                  title=f"Displacement LOS for point {point_id}",
                  markers=True, 
                  labels={'displacement': 'Displacement[mm]'})

    fig.add_scatter(x=filtered_data['timestamp'], y=filtered_data['displacement'], 
                    mode='lines+markers', 
                    name='InSAR measured displacement', 
                    line=dict(color='blue'))

    if not filtered_last_n_data.empty:
        if 'predicted_value' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['predicted_value'], 
                            mode='lines+markers', 
                            name='Predicted Displacement', 
                            line=dict(color='orange'))

        if 'upper_bound' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['upper_bound'], 
                            mode='lines', line=dict(color='yellow', dash='dash'),
                            name='Upper Bound p=95')

            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['lower_bound'],
                            mode='lines', line=dict(color='yellow', dash='dash'),
                            fill='tonexty', fillcolor='rgba(255, 252, 127, 0.2)',
                            name='Lower Bound p=95')

        anomalies_95 = filtered_last_n_data[filtered_last_n_data['is_anomaly'] == 1]
        if not anomalies_95.empty:
            fig.add_scatter(x=anomalies_95.index, y=anomalies_95['displacement'], 
                            mode='markers', name='Anomalies p=95', 
                            marker=dict(color='yellow', size=10))

        if 'upper_bound_99' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['upper_bound_99'], 
                            mode='lines', line=dict(color='red', dash='dash'),
                            name='Upper Bound p=99')

            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['lower_bound_99'],
                            mode='lines', line=dict(color='red', dash='dash'),
                            fill='tonexty', fillcolor='rgba(254, 121, 104, 0.1)',
                            name='Lower Bound p=99')

            anomalies_99 = filtered_last_n_data[filtered_last_n_data['is_anomaly_99'] == 1]
            if not anomalies_99.empty:
                fig.add_scatter(x=anomalies_99.index, y=anomalies_99['displacement'], 
                                mode='markers', name='Anomalies p=99', 
                                marker=dict(color='red', size=10))

    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Displacement LOS[mm]', 
        legend_title="Legend",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.05)
    )

    return fig, {'display': 'block'}
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
