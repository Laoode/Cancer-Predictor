<h1 align="center">🦠 The Breast Cancer Diagnosis Predictor App 🦠</h1>

<h2 align="left">🔽 Overview</h2>
<p>Breast cancer is the most common cancer among women such that the existence of a precise and reliable system for the diagnosis of benign or malignant tumors is critical. Nowadays, using machine learning techniques, detection and early diagnosis of this cancer can be done with greater accuracy. The method used in this project is logistic regression, which is a supervised learning method. To select or delete a feature, feature weighting is used. In logistic regression, the Sigmoid function is used for classification, which ensures that the output is in the range [0–1]. Simulation results show that the proposed method reaches accuracy 97.3% of our model</p>
<p>The Breast Cancer Diagnosis app is a machine learning-powered tool designed to assist medical professionals in diagnosing breast cancer. Using a set of measurements, the app predicts whether a breast mass is benign or malignant. It provides a visual representation of the input data using a radar chart and displays the predicted diagnosis and probability of being benign or malignant. The app can be used by manually inputting the measurements or by connecting it to a cytology lab to obtain the data directly from a machine. The connection to the laboratory machine is not a part of the app itself.</p>

A live version of the application can be found on [Streamlit Community Cloud](https://cancer-predictor-6fp4fsjku2yujwbalreskz.streamlit.app/).

<h2 align="left">📶 Dashboard</h2>
<img src="https://github.com/Laoode/Cancer-Predictor/blob/main/Images/dashboard.png" alt="logo">


<h2 align="left">🔃 Installation</h2> 
<p>You can run this inside a virtual environment to make it easier to manage dependencies. I recommend using `conda` to create a new environment and install the required packages. You can create a new environment called `breast-cancer-predictor` by running:</p>

```bash
conda create -n breast-cancer-predictor python=3.11
```
<p>Then, activate the environment:</p>

```bash
conda activate breast-cancer-predictor
```
To install the required packages, run:
```bash
pip install -r requirements.txt
```
<p>This will install all the necessary dependencies, including numpy, streamlit, sckit-learn, pandas, matplotlib</p>

<h2 align="left">🌐 Usage</h2> 

To start the app, simply run the following command:
```bash
streamlit run app/main.py
```
<p>After running the command, Streamlit will automatically launch the app in your default web browser.</p>
