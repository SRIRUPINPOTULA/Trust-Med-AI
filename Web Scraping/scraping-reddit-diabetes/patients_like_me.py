import requests
from bs4 import BeautifulSoup

url = "https://www.patientslikeme.com/conditions/diabetes-type-1"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Example: extract page title and meta description
    title = soup.find("title").text.strip()
    description = soup.find("meta", {"name": "description"})
    description = description["content"] if description else "No description found."

    print("Page Title:", title)
    print("Description:", description)
else:
    print("Failed to fetch page:", response.status_code)
