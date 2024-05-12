import urllib.parse as urlparse
import imageio
import dash
import cv2
import time
import random
from pytube import YouTube
import plotly.graph_objs as go
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
import base64
from dash.long_callback import DiskcacheLongCallbackManager
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from engine.PoseEstimation import poseDetector
import os

CARD_STYLE = "https://fonts.googleapis.com/css?family=Saira+Semi+Condensed:300,400,700"

## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = dash.Dash(
    __name__,
    external_stylesheets=[CARD_STYLE],
    long_callback_manager=long_callback_manager,
    suppress_callback_exceptions=True,
)

app.title = 'Posture Analyzer'

def angle_graph(list1, list2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(0, len(list2))], y=list2, mode='lines', name='Shoulder'))
    fig.add_trace(go.Scatter(x=[i for i in range(0, len(list1))], y=list1, mode='lines', name='Hip'))
    fig.update_layout(title='Body Joints Angle Over Time', template='plotly_white', xaxis_title='Time', yaxis_title='Angle (degrees)', height=370, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', family='changa'))
    return fig

model_input = html.Div([
    dmc.Col(
        dmc.Text("Model Settings", color='white', style={"fontSize": 20, "font-family":'changa'})
    ),
    dmc.Center([
        dmc.Col([
            dmc.Text('Start Time (s): ', color='white', style={"font-family":'changa'}),
            dmc.NumberInput(
                id='starttime-input',
                label="",
                value=0,
                min=0,
                style={"color": 'white', "font-family":'changa'},
                className='mantine-1w07r5r'
            )
        ], span=10)
    ]),
    dmc.Center([
        dmc.Col([
            dmc.Text('Duration (s): ', color='white', style={"font-family":'changa'}),
            dmc.NumberInput(
                id='duration-input',
                label="",
                value=5,
                min=1,
                max=10,
                style={"color": 'white', "font-family":'changa'},
                className='mantine-1w07r5r'
            )
        ], span=10)
    ]),
    dmc.Center([
        dmc.Col([
            dmc.Text('Detection Confidence: ', color='white', style={"font-family":'changa'}),
            dmc.Slider(
                id="detection-slider",
                value=5,
                min=0, max=10, step=0.5,
                style={"width": 250},
            )
        ], span=10)
    ]),
    dmc.Center([
        dmc.Col([
            dmc.Text('Tracking Confidence: ', color='white', style={"font-family":'changa'}),
            dmc.Slider(
                id="tracking-slider",
                value=5,
                min=0, max=10, step=0.5,
                style={"width": 250},
            )
        ], span=10)
    ]),
    dmc.Space(h=20),
], style={'border':'1px solid white','overflowX': 'hidden'})

export_card = html.Div([
    dmc.Col(
        dmc.Text("Export Output", color='white', style={"fontSize": 20, "font-family":'changa'})
    ),
    dmc.Center([
        dmc.Col([
            html.Br(),
            html.Br(),
            dmc.Center(DashIconify(icon="el:download", width=90, color='green')),
            html.Br(),
            html.Br(),
            dmc.Center(dmc.Button('Download GIF', id='download-button', style={"font-family":'changa'})),
            dcc.Download(id='download-image')
        ], span=10)
    ]),
], style={'border':'1px solid white', 'height':'380px','overflowX': 'hidden'})

output_body = html.Div([
    dmc.Grid([
        dmc.Col([
            export_card
        ], span=3),
        dmc.Col([
            dmc.LoadingOverlay(dmc.Image(id='model-output', height=380, style={'max-width':'810'}), loaderProps={"variant": "bars", "size": "xl"})
        ], span=6),
        dmc.Col([
            dcc.Graph(id='line-graph', config={'displaylogo':False}, style={'height':375})
        ], span=3, style={'border':'1px solid white'}),
    ])
], id='output-div', style={"visibility": "hidden",'overflowX': 'hidden'})


app.layout = dmc.Container([
    dmc.Center([
        dmc.Grid([
            dmc.Col([
                dmc.Text(
                    "Full Body Posture Analysis App",
                    color='white',
                    align="center",
                    weight=500,
                    style={"fontSize": 36, "font-family":'changa'}
                )
            ], span=12)
        ]),
    ]),
    html.Hr(),
    dmc.Space(h=30),
    dmc.Grid([
        dmc.Col([
            html.Div([
                dcc.Upload(
                    id='upload-video',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                        "font-family":'changa',
                        'color':'white'
                    },
                    multiple=False  # Changed to False to allow only one file upload
                ),
                html.Div(id='output-video')
            ])
        ], span=12)
    ]),
    dmc.Space(h=30),
    dmc.Center(
        dmc.Button("Run Model", id="run-model-button", variant='filled', size='lg', style={"font-family":'changa'})
    ),
    dmc.Space(h=30),
    dcc.Store(id='output-path'),
    model_input,
    dmc.LoadingOverlay(output_body, loaderProps={"variant": "bars", "size": "xl"}),
    dmc.Space(h=30)
], fluid=True, style={'backgroundColor':'#111b2b','overflow-y':'hidden','overflowX': 'hidden'})


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    with open(filename, 'wb') as f:
        f.write(decoded)
    return html.Video(src=contents, controls=True)

@app.callback(Output('output-video', 'children'),
              Input('upload-video', 'contents'),
              State('upload-video', 'filename'))

def update_output(contents, filename):
    if contents is not None:
        children = parse_contents(contents, filename)
        return children

from dash.exceptions import PreventUpdate

# Global variable to store the last processed filename
last_processed_filename = None

# Callback to update start time and duration
@app.callback(
    Output("starttime-input", "value"),
    Output("duration-input", "value"),
    Input("upload-video", "contents"),
    State("upload-video", "filename"),
    prevent_initial_call=True
)
def update_time(contents, filename):
    global last_processed_filename

    # Check if filename has changed
    if filename != last_processed_filename:
        last_processed_filename = filename

        if contents is not None:
            # Fetch the length of the video and calculate the duration
            content_type, content_string = contents.split(',')[0], contents.split(',')[1]
            decoded = base64.b64decode(content_string)
            video_path = 'temp_video.mp4'  # Temporarily save the video to calculate its length
            with open(video_path, 'wb') as f:
                f.write(decoded)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            # Release the video capture object
            cap.release()

            # Cleanup: Delete the temporary video file
            os.remove(video_path)

            return 0, duration

    # Return default values or PreventUpdate when no video is uploaded or no change in contents
    if contents is None:
        raise PreventUpdate
    else:
        return dash.no_update, dash.no_update
    
# Download GIF
@app.callback(
    Output("download-image", "data"),
    Input("download-button", "n_clicks"),
    State("output-path", "data"),
    prevent_initial_call=True
)
def func(n_clicks, path):
    if n_clicks:
        return dcc.send_file(path)

# running model
@app.long_callback(
    Output("model-output", "src"),
    Output("output-path", "data"),
    Output("line-graph", "figure"),
    Input("run-model-button", "n_clicks"),
    State("upload-video", "contents"),
    State("upload-video", "filename"),
    State("starttime-input", "value"),
    State("duration-input", "value"),
    State("detection-slider", "value"),
    State("tracking-slider", "value"),
    running=[
            (Output("run-model-button", "disabled"), True, False),
            (Output("run-model-button", "children"), "Running Model", "Run Model"),
            (Output("output-div", "style"), {"visibility": "hidden"}, {"visibility": "visible"})
        ],
    manager=long_callback_manager,
    prevent_initial_call=True
)
def show_output(n_clicks, contents, filename, start, duration, detectionCon, trackingCon, save_directory="./uploaded_videos"):
    if n_clicks and contents is not None:
        video_data = contents.split(',')[1]
        video_data = base64.b64decode(video_data)
        video_path = filename  # Removed os.path.join() as it's already the filename
        
        # Write the video data to the specified file
        with open(video_path, "wb") as f:
            f.write(video_data)
        
        frames = []
        angle_list1 = []
        angle_list2 = []

        # Create the output directory if it doesn't exist
        output_directory = "./PostureTracker/assets/model_runtime_output"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Read the video file and process frames
        cap = cv2.VideoCapture(video_path)
        milliseconds = 1000
        end_time = start + duration
        cap.set(cv2.CAP_PROP_POS_MSEC, start * milliseconds)
        pTime = 0
        detector = poseDetector(detectionCon=detectionCon/10, trackCon=trackingCon/10)
        # Loop through video frames and process them
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * milliseconds:
            success, img = cap.read()
            if not success:
                break  # Break loop if no frame is retrieved
            
            # Perform pose detection and angle calculation here
            # Example:
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                angle1 = detector.findAngle(img, 28, 24, 27)
                angle_list1.append(angle1)
                angle2 = detector.findAngle(img, 14, 12, 24)
                angle_list2.append(angle2)
            
            # Add the frame to the list
            frames.append(img)

        cap.release()

        # Check if any frames were processed
        if not frames:
            return dash.no_update, dash.no_update, dash.no_update

        # Save processed frames as GIF
        filename = str(random.random())
        gif_path = os.path.join(output_directory, f"output{filename}.gif").replace("\\", "/")
        with imageio.get_writer(gif_path, mode="I") as writer:
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_frame)

        # Generate angle graph
        fig = angle_graph(angle_list1, angle_list2)

        return gif_path, gif_path, fig  # Path for download and path for displaying GIF

    else:
        # Handle the case when no video is uploaded or button is not clicked
        return dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
    app.run_server()
