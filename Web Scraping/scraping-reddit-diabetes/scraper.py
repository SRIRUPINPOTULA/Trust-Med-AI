import requests
import csv
import time

input_csv = "diabetes_post_links.csv"
output_csv = "diabetes_post_texts2.csv"

headers = {"User-Agent": "reddit-scraper/1.0 (by u/yourusername)"}

# Open the input CSV
with open(input_csv, "r", encoding="utf-8") as infile, open(output_csv, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["Title", "Author", "URL", "Full_Text"])  # Output header

    for i, row in enumerate(reader, start=74):
        url = row["URL"].strip()
        json_url = url.rstrip("/") + "/.json"  # Reddit JSON endpoint

        try:
            response = requests.get(json_url, headers=headers)
            if response.status_code != 200:
                print(f"⚠️ ({i}) Skipping {url} - HTTP {response.status_code}")
                continue

            data = response.json()
            post_data = data[0]["data"]["children"][0]["data"]

            title = post_data.get("title", "")
            author = post_data.get("author", "")
            selftext = post_data.get("selftext", "").replace("\n", " ").strip()

            # Write to new CSV
            writer.writerow([title, author, url, selftext])
            print(f"✅ ({i}) Scraped: {title[:50]}...")

            # polite delay
            time.sleep(2)

        except Exception as e:
            print(f"❌ ({i}) Error scraping {url}: {e}")
            continue

print(f"\n✅ Done! Full post texts saved to '{output_csv}'.")
