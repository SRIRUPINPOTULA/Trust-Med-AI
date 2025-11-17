import requests
from bs4 import BeautifulSoup
import csv
import time

url = "https://www.webmd.com/diabetes/default.htm"
headers = {"User-Agent": "my-scraper/0.1 (by u/yourusername)"}

resp = requests.get(url, headers=headers)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

output_file = "webmd_diabetes_links.csv"
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Link Text", "URL"])

    # Find all anchor tags within the diabetes section
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # focus on internal links to WebMD /diabetes
        if href.startswith("/diabetes/") or "webmd.com/diabetes" in href:
            full_url = href if href.startswith("http") else "https://www.webmd.com" + href
            text = a.get_text().strip()
            if text:
                writer.writerow([text, full_url])

print(f"✅ Done! Links saved to '{output_file}'")
