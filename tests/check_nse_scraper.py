"""Test NSE scraping using requests + BeautifulSoup."""
import requests
from bs4 import BeautifulSoup

url = "https://live.mystocks.co.ke"

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Failed to fetch {url}: {e}")
    exit(1)

soup = BeautifulSoup(response.text, 'html.parser')
tables = soup.find_all("table")

if tables:
    print(tables[0].prettify()[:500])  # Print first 500 chars of the first table
else:
    print("No tables found on the page.")
    print("Page preview:")
    print(soup.get_text()[:500])
