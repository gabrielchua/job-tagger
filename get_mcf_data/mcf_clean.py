import pandas
import re

df = pd.read_csv("../data/mcf_jobs.csv")

def clean_html(text):
    # Remove HTML tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    
    # Replace HTML escape entities with their characters
    cleantext = cleantext.replace('&amp;', '&')
    cleantext = cleantext.replace('&lt;', '<')
    cleantext = cleantext.replace('&gt;', '>')
    cleantext = cleantext.replace('&quot;', '"')
    cleantext = cleantext.replace('&#39;', "'")
    
    # Remove line breaks
    cleantext = cleantext.replace('\n', ' ')  # Replace with space. If you prefer no space, replace with ''
    
    # Remove full HTTP links
    cleantext = re.sub(r'http\S+', '', cleantext)
    
    return cleantext

df = df[["title", "description"]]
df.loc[:, 'title'] = df.loc[:, 'title'].apply(clean_html)
df.loc[:, 'description'] = df.loc[:, 'description'].apply(clean_html)

df.to_csv("../data/mcf_jobs_clean.csv", index=False)
