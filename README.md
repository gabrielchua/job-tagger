# Job Tagger Application 🤖

This Streamlit application is designed to assist in tagging job postings. Users are presented with a random job posting from the MCF dataset, and they're asked to tag the posting with the most suitable job category from the SSOC (Singapore Standard Occupational Classification) list.

## Features

1. **Random Job Selection**: Each session will randomly select a job posting from the MCF dataset.
2. **SSOC Predictions**: Based on a pre-trained model, the application suggests the top-n SSOC categories that might match the job posting.
3. **User Interaction**: Users can choose the most suitable SSOC category or skip if none of the predictions match.
4. **Integration with Google Sheets**: Selected SSOC categories, along with the job posting, are saved in a Google Sheet for future reference.

## Setup

1. Ensure you have the required libraries installed:
    - streamlit
    - pandas
    - numpy
    - pickle
    - datetime

2. The following datasets and models should be present in the respective directories:
    - Pre-trained model: `model/autocoder.pkl`
    - SSOC list: `data/ssoc.csv`
    - MCF jobs list: `data/mcf_jobs_clean.csv`

3. If you wish to connect with Google Sheets API, make sure you've set up the API and the `setup_gspread` function in the `utils` file is configured correctly.

## How to Run

Navigate to the directory containing the Streamlit script and run:

```
streamlit run app.py
```

## UI Overview

- **Title**: At the top, you'll see the title "Help us tag jobs 🙏".
- **Instructions**: A dropdown expander contains the instructions for users.
- **Job Display**: The randomly selected job posting is displayed on the left column.
- **SSOC Predictions**: On the right column, top-n SSOC predictions are displayed as buttons. Users can click on the category that best matches the job posting.
- **Skip Button**: If none of the predictions are satisfactory, users can skip tagging by pressing the "❌ None of the above/SKIP ❌" button.
- **Category Descriptions**: A toggle at the bottom allows users to view descriptions of the predicted SSOC categories.