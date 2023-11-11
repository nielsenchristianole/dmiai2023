# print(">>> Importing libraries...")
# from datasets import load_dataset

# print(">>> Loading dataset...")
# dataset = load_dataset("wikipedia", "20220301.da", split="train", beam_runner="DirectRunner")

# print(dataset[0])   

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def clean_text(text):
    # Remove all non-alphanumeric characters and all special unicode characters
    # Keep: .,;:!?-$%&/'" and danish characters along with spaces
    text = re.sub(r'[^a-zA-Z0-9æøåÆØÅ.,;:!? \-$/\'"%&]', ' ', text)
    
    # Remove all double spaces and links
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    
    return text

def extract_data(urls, min_length = 30, max_length = 1000):
    extracted = []
    
    skipped_bodies = 0
    skipped_articles = 0
    
    for url in tqdm(urls):
        # Send a HTTP request to the URL of the webpage
        response = requests.get(url)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        content = {'header': [], 'summary': [], 'body': []}
        
        ### EXTRACT HEADERS ###
        heads = soup.find_all('h1', class_='article__title headline headline--mega') # politiken
        
        div_temp = soup.find('div', class_='article__header') # bt
        heads += div_temp.find_all('h1') if div_temp is not None else [] # bt
        
        div_temp = soup.find('div', class_='header-elements') # information
        heads += div_temp.find_all('h1') if div_temp is not None else [] # information
        
        heads += soup.find_all('h1', class_='article-header__title') # berlinske
        
        ### EXTRACT SUMMARIES ###
        summaries = soup.find_all(class_='summary__p') # politiken
        
        summaries += soup.find_all('div', class_='field field-name-field-underrubrik') # information
        
        summaries += soup.find_all('p', class_='article-header__intro') # berlinske
        
        ### EXTRACT BODIES ###
        bodies = soup.find_all(class_='body__p') # politiken
        
        div_temp = soup.find('div', class_='article-content') # bt
        bodies += div_temp.find_all('p') if div_temp is not None else [] # bt
        
        div_temp = soup.find('div', class_='field field-name-body') # information
        bodies += div_temp.find_all('p') if div_temp is not None else [] # information
        
        div_temp = soup.find('div', class_='article-body') # berlinske
        bodies += div_temp.find_all('p') if div_temp is not None else [] # berlinske
        
        
        # For each header, summary and body, extract the text and store it in the dictionary
        for header in heads:
            content['header'].append(clean_text(header.text))
        for summary in summaries:
            content['summary'].append(clean_text(summary.text))
        for body in bodies:
            if len(body.text.split()) < min_length or len(body.text.split()) > max_length:
                print("Skipping body [length not in range]")
                skipped_bodies += 1
                continue
            content['body'].append(clean_text(body.text))
            
        if content['header'] == [] or content['body'] == []:
            print("Skipping [missing header or body]:", url[:10])
            skipped_articles += 1
            continue    
        
        extracted.append(content)

    # Save the extracted content as a .npz file
    np.savez('ds_constructor/dataset.npz', extracted=extracted)
    
    print("Skipped bodies:", skipped_bodies)
    print("Skipped articles:", skipped_articles)
    
    print("Extraction complete!")
    
def get_urls(path = 'ds_constructor/websites.txt')
    # Read all URLs from websites.txt (each line is a URL)
    with open(path, 'r', encoding='utf-8') as f:
        urls = f.readlines()
        
    # Remove duplicates
    urls = list(set(urls))
    
    return urls
    

# Read the dataset.npz file and count the number of headlines, summaries and bodies
dataset = np.load('ds_constructor/dataset.npz', allow_pickle=True)['extracted']
print("Number of headlines:", sum([len(d['header']) for d in dataset]))
print("Number of summaries:", sum([len(d['summary']) for d in dataset]))
print("Number of bodies:", sum([len(d['body']) for d in dataset]))

# Get length distribution of all individual segments in all bodies
lengths_scrape = []
for d in dataset:
    for body in d['body']:
        lengths_scrape.append(len(body.split()))
        if len(body.split()) < 30:
            print(body)

# load data.csv from data
data = pd.read_csv('data/data.csv')

# Get the length of each datapoint in dataset['text']
lengths_example = []
for text in data['text']:
    lengths_example.append(len(text.split()))

# Print max, min, mean and median length
print("=== Example dataset ===")
print("Max length:", max(lengths_scrape))
print("Min length:", min(lengths_scrape))
print("Mean length:", np.mean(lengths_scrape))
print("Median length:", np.median(lengths_scrape))

print("\n=== Scraped dataset ===")
print("Max length:", max(lengths_example))
print("Min length:", min(lengths_example))
print("Mean length:", np.mean(lengths_example))
print("Median length:", np.median(lengths_example))

# from openai import OpenAI

# client = OpenAI(api_key=)

# models = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
# model_idx = 1

# people = [
#     "Du er en meget professionel og vel formuleret journalist med mange års erfaring.",
#     "Du er en dygtig journalist med medium erfaring.",
#     "Du er en nyansat journalist med lidt erfaring.",
# ]

# content = [
#     " Du skriver artikler om teknologi og dens indflydelse på samfundet",
#     " Du skriver politisk prægede artikler om teknologi og dens indflydelse på samfundet",
#     " Du har meget viden om teknologi og menneskerne bag hvilket er focuspunktet i dine artikler",
#     " Du elsker at quote folk inden for tech industrien i dine artikler",
#     " Du skriver mange artikler om chatgpt og ai",
#     " Du skriver artikler om teknologi og dens inflydelse men er meget kritisk over teknologien",
# ]

# length = ["1 til 5", "1 til 10", "2 til 3", "3 til 7"]

# word = []

# # Pick a random person and content
# person = np.random.choice(people)
# content = np.random.choice(content)
# length = np.random.choice(length)
# word = np.random.choice(word)
# headline = ""

# completion = client.chat.completions.create(
#   model=models[model_idx],
#   messages=[
#     {"role": "system", "content": person + content},
#     {"role": "user", "content": f"Skriv 10 små udpluk fra dine danske artikler om teknologi. Specielt kunstig intelligens, chatbots (chatgpt) og machine learning. \
#                                  Hvert udpluk skal ikke (nødvendigvis) komme fra samme artikel og hvert udpluk bør ikke være sammenhængende.   \
#                                  Includer eventuelt et quote fra en bruger eller en ekspert. Du må gerne inkludere specifikke detaljer. \
#                                  Hvert udpluk skal være mellem {length} sætninger. Nogle af dem må godt være meget korte men varier længden\
#                                  Overvej også inkluder udpluk der virker totalt ude af kontekst.\
#                                  Brug dobbelt linje mellem hvert udpluk.\
#                                  Inkluder ordet {word} i mindst et af udplukkene.\  
#                                  Overskriften på artiklen er: {headline}"}
#   ]
# )

# print(completion.choices[0].message.content)