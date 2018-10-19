# Scrape WAV files from Toronto emotional speech set (TESS)

import requests
from bs4 import BeautifulSoup
import urllib3


BASE_URL = 'https://tspace.library.utoronto.ca'
URL = 'https://tspace.library.utoronto.ca/handle/1807/24487'

http = urllib3.PoolManager()

page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
links = soup.find_all('a', href=True)
data_links = [link for link in links if 'Toronto emotional speech set (TESS)' in link.text]
for link in data_links:
    data_url = BASE_URL + link['href']
    data_page = requests.get(data_url)
    soup = BeautifulSoup(data_page.content, 'html.parser')
    wavs = soup.find_all('a', href=True, target='_blank')
    for wav in wavs:
        name = wav.text
        req = http.request('GET', BASE_URL + wav['href'], preload_content=False)
        res = req.read()
        f = open('/Users/kailinlu/Documents/emotional-speech/data/' + name, 'wb')
        f.write(res)
        f.close()

