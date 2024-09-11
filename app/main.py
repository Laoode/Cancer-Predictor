import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import base64
import os

st.set_page_config( 
    page_title='Breast Cancer Predictor',
    page_icon='"👩‍⚕️',
    layout='wide',
    initial_sidebar_state='expanded'
)
@st.cache_data
def get_img_as_base64(file):
    file_path = os.path.join(os.path.dirname(__file__), file)
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

img = get_img_as_base64("../assets/bckgr.png")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover; 
background-position: center; 
background-repeat: no-repeat; 
background-attachment: fixed; 
height: 100vh;
margin: 0;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
def load_csv(file):
    file_path = os.path.join(os.path.dirname(__file__), file)
    return pd.read_csv(file_path)

data = load_csv("../model/processed_data.csv")

def sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    slider_labels =[
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]
    
    input_dict ={}
    
    for label, key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label, 
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data_scaled= data.copy()
    
    X = data_scaled.drop(['diagnosis'], axis = 1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value
        
    return scaled_dict
    

def get_radar_chart(input_data):
    st.markdown(
    """
    <style>
        /* Membungkus SVG dengan elemen yang dapat menggunakan backdrop-filter */
        .js-plotly-plot .plot-container {
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.1); /* Transparansi latar belakang */
            border-radius: 10px; /* Sudut membulat jika diinginkan */
            padding: 10px; /* Spasi dalam elemen */
        }

        .js-plotly-plot .plot-container .main-svg {
            background-color: transparent !important; /* Menjadikan background SVG transparan */
        }
        .js-plotly-plot .legend .bg {
            fill: rgba(0, 0, 0, 0) !important;
        }
        .js-plotly-plot .plot-container .main-svg .polarlayer {
            transform: translate(60px, 0) !important;
        }
        .js-plotly-plot .legend {
            transform: translate(10px, 10px) !important;
        }
        .js-plotly-plot .modebar-container {
            background-color: transparent !important;
            right: 10px !important; /* Adjust as needed */
            top: 10px !important; /* Adjust as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="glass-effect">
            <!-- Di sini adalah tempat untuk menampilkan grafik SVG atau konten lainnya -->
        </div>
        """,
        unsafe_allow_html=True
    )
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()
    
    input_data=get_scaled_values(input_data)
    
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'], input_data['smoothness_mean'],
            input_data['compactness_mean'], input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'], input_data['smoothness_se'],
            input_data['compactness_se'], input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'], input_data['area_worst'], input_data['smoothness_worst'],
            input_data['compactness_worst'], input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    return fig

def format_probability(prob):
    return f"{prob:.10f}"
def load_pickle(file):
    file_path = os.path.join(os.path.dirname(__file__), file)
    return pickle.load(open(file_path, "rb"))

def add_predictions(input_data):
    st.markdown("""
    <style>
    .probability-display {
        background-color: rgba(0, 0, 0, 0.7);
        color: #3498db;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    model = load_pickle("../model/model.pkl")
    scaler = load_pickle("../model/scaler.pkl")
    
    input_array = np.array(list(input_data.values())).reshape(1,-1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    probabilities = model.predict_proba(input_array_scaled)[0]
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis bright-green'>Benign</span>",
                 unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Malignant</span>",
                 unsafe_allow_html=True)
    
    st.markdown(f"""
    <p>Probability of being benign: 
        <span class="probability-display">{format_probability(probabilities[0])}</span>
    </p>
    <p>Probability of being malicious: 
        <span class="probability-display">{format_probability(probabilities[1])}</span>
    </p>
    """, unsafe_allow_html=True)

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
def main():
    input_data=sidebar()
    css_file_path = os.path.join(os.path.dirname(__file__), "../assets/style.css")
    with open(css_file_path) as f:
        st.markdown("<style>{}<style/>".format(f.read()), unsafe_allow_html=True)
    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.')
    
    col1, col2 = st.columns([4,1])
    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
if __name__ == '__main__':
    main()