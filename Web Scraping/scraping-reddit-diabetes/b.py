import requests
from bs4 import BeautifulSoup
import csv
import time

# Input CSV (from your previous step)
input_csv = "webmd_diabetes_links.csv"
# Output CSV for scraped content
output_csv = "webmd_diabetes_articles.csv"

headers = {"User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0; +https://github.com/yourusername)"}

with open(input_csv, "r", encoding="utf-8") as infile, open(output_csv, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["Title", "URL", "Content"])  # output columns

    for i, row in enumerate(reader, start=1):
        url = row["URL"].strip()
        if not url.startswith("http"):
            url = "https://www.webmd.com" + url

        print(f"📄 ({i}) Scraping: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"⚠️ ({i}) Skipping — HTTP {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract article title
            title = soup.find("title").get_text(strip=True) if soup.find("title") else "No title"

            # Extract main article body text
            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            # Clean up whitespace
            content = " ".join(content.split())

            writer.writerow([title, url, content])
            print(f"✅ ({i}) Done: {title[:60]}...")
            time.sleep(2)  # polite delay

        except Exception as e:
            print(f"❌ ({i}) Error scraping {url}: {e}")
            continue

print(f"\n🎉 Finished scraping! All articles saved to '{output_csv}'.")
