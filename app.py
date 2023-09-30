# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pickle
from ssoc_autocoder.combined_model import SSOCAutoCoder
from utils import append_to_sheet, setup_gspread, st_button_css, reset_app, random_job

# Set page configuration
st.set_page_config(layout="wide", page_title="Job Tagger", page_icon=":robot_face:")

# Configuration
top_n = 3

# Basic CSS
customized_button = st_button_css()

# Set CSS for h1 tag
st.markdown("""
    <style>
        h1 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialise page
if "model" not in st.session_state:
    # Load the pre-trained model
    with open("model/autocoder.pkl", "rb") as inp:
        st.session_state["model"] = pickle.load(inp)
        print(st.session_state["model"].MAX_TOKEN_LENGTH)

if "ssoc_list" not in st.session_state:
    # Load the SSOC list
    st.session_state["ssoc_list"] = pd.read_csv("data/ssoc.csv")

if "mcf_jobs" not in st.session_state:
    # Load the MCF jobs list
    st.session_state["mcf_jobs"] = pd.read_csv("data/mcf_jobs_clean.csv")

# Set up Google Sheets API
google_sheet = setup_gspread()

# Get random job title and description
if ("job_title" not in st.session_state) or (st.session_state["job_title"] is None):
    st.session_state["job_title"], st.session_state["job_desc"] = random_job(st.session_state["mcf_jobs"])

if len(st.session_state["job_desc"]) > 1000:
    st.session_state["job_desc"] = st.session_state["job_desc"][:1000] + "..."

# Get predictions
if ("prediction" not in st.session_state) or (st.session_state["prediction"] is None):
    # Make predictions using the pre-trained model
    st.session_state["prediction"] = st.session_state["model"].predict(st.session_state["job_title"], st.session_state["job_desc"], top_n)

possible_ssoc = st.session_state["prediction"]["prediction"]
possible_ssoc_conf = st.session_state["prediction"]["confidence"]

# Main UI
st.title("Help us tag jobs 🙏")

with st.expander("Instructions"):
    st.write("This application will randomly select a job posting from MCF. Please select the most representative category.")

col1, col2 = st.columns(2)

with col1:
    st.success(f"""**{st.session_state["job_title"]}:** 

{st.session_state["job_desc"]}""")

with col2:
    st.divider()
    st.markdown("""
    <style>
        .center-text {
            text-align: center;
        }
    </style>
    <div class="center-text"><b>Please select the most suitable category</b></div>
""", unsafe_allow_html=True)
    st.write("")

    ssoc_label_list = []
    ssoc_desc_list = []

    for i in range(0, top_n):
        predicted_ssoc = int(possible_ssoc[i])
        predicted_confidence = float(possible_ssoc_conf[i])

        match_ssoc = list(st.session_state["ssoc_list"]["ssoc"] == predicted_ssoc)

        predicted_ssoc_label = st.session_state["ssoc_list"].loc[match_ssoc, "title"].tolist()
        ssoc_label_list.append(predicted_ssoc_label)

        predicted_ssoc_desc = st.session_state["ssoc_list"].loc[match_ssoc, "description"].tolist()
        ssoc_desc_list.append(predicted_ssoc_desc)

        if st.button(predicted_ssoc_label[0]):
            # Append the selected job to the Google Sheets
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')
            append_to_sheet(google_sheet, current_date, current_time, st.session_state["job_title"], st.session_state["job_desc"], possible_ssoc[i], possible_ssoc_conf[0][:5], possible_ssoc[0], possible_ssoc_conf[0][:5])
            reset_app()
        
    if st.button("❌ None of the above/SKIP ❌"):
        append_to_sheet(google_sheet, current_date, current_time, st.session_state["job_title"], st.session_state["job_desc"], "NIL", "NIL", possible_ssoc[0], possible_ssoc_conf[0][:5])


st.divider()

# Toggle to show category descriptions
show = st.toggle("Show Category Descriptions")

if show:
    for i in range(0, top_n):
        st.markdown(f"**{ssoc_label_list[i]}** : {ssoc_desc_list[i]}")