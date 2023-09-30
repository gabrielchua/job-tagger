import streamlit as st
from google.oauth2 import service_account
import gspread
import urllib3
import json
import re
import pandas as pd


def setup_gspread():
    """
    Sets up the connection to a Google Sheets document using the Google Sheets API.

    Returns:
    gspread.Worksheet: The first worksheet of the Google Sheets document.
    """
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(credentials)
    sh = gc.open_by_url(st.secrets["private_gsheets_url"])
    worksheet = sh.get_worksheet(0) # Assuming you want to write to the first sheet
    return worksheet

# Append data to the Google Sheet
def append_to_sheet(worksheet, current_date, current_time, job_title, job_desc, chosen_ssoc, chosen_ssoc_conf, model_ssoc, model_ssoc_conf):
    worksheet.append_row([current_date, current_time, job_title, job_desc, chosen_ssoc, chosen_ssoc_conf, model_ssoc, model_ssoc_conf])

def st_button_css():
    return st.markdown("""
    <style>
    .stDownloadButton, div.stButton {text-align:center;}
    .stDownloadButton button, div.stButton > button:first-child {
        width: 100%;   <!-- This line sets the width to 100% -->
    }
    
    </style>
""", unsafe_allow_html=True)

def random_job(job: pd.DataFrame) -> tuple:
    """
    Returns a random job title and description from the given DataFrame.

    Parameters:
    job (pd.DataFrame): A pandas DataFrame containing job titles and descriptions.

    Returns:
    tuple: A tuple containing a random job title and description.
    """
    random_row = job.sample()
    title = random_row["title"].values[0]
    description = random_row["description"].values[0]
    return title, description

def reset_app():
    st.session_state["prediction"] = None
    st.session_state["job_title"] = None
    st.session_state["job_desc"] = None
    st.rerun()