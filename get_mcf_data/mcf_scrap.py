import os
import requests
import csv
import time
import sys  # Import the sys module for command-line arguments

# Check if the total_count argument is provided
if len(sys.argv) < 2:
    print("Please provide total_count as an argument.")
    sys.exit(1)  # Exit the script

try:
    total_count = int(sys.argv[1])  # Convert the argument to an integer
except ValueError:
    print("Invalid total_count argument. Please provide a valid number.")
    sys.exit(1)

BASE_URL = 'https://api.mycareersfuture.gov.sg/v2/jobs/'
LIMIT = 100  # Number of items per page

# Ensure 'data' directory exists; if not, create it
if not os.path.exists('data'):
    os.makedirs('data')

# Now adjust the CSV_FILE path
CSV_FILE = '../data/mcf_jobs.csv'


########
total_pages = (total_count // LIMIT) + (1 if total_count % LIMIT else 0)

all_data = []

for page_num in range(total_pages):
    url = f"{BASE_URL}?limit={LIMIT}&offset={page_num * LIMIT}"
    response = requests.get(url)
    response.raise_for_status()
    
    # Extract data from the 'results' key based on the given example
    data_page = response.json()['results']
    all_data.extend(data_page)
    
    # Pause every 500 queries
    if (page_num + 1) % 5 == 0:
        time.sleep(5)

# Ensure there's data and it's in dictionary format before writing to CSV
if all_data and isinstance(all_data[0], dict):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)
        print(f"Data written to {CSV_FILE}")
else:
    print("No data to write or data is not in expected format.")
