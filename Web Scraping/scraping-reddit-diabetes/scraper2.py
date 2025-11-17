import requests
import csv
import time

# Input and output files
input_csv = "diabetes_post_links.csv"       # Your CSV with Reddit post URLs
output_file = "diabetes_posts_text.txt"     # Single text file for all posts

headers = {"User-Agent": "reddit-scraper/1.0 (by u/yourusername)"}

with open(input_csv, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)

    outfile.write(f"Reddit Scrape: r/diabetes\n")
    outfile.write("=" * 100 + "\n\n")

    for i, row in enumerate(reader, start=27):
        url = row["URL"].strip()
        json_url = url.rstrip("/") + "/.json"  # Reddit JSON endpoint

        try:
            resp = requests.get(json_url, headers=headers)
            if resp.status_code != 200:
                print(f"⚠️ ({i}) Skipping {url} - HTTP {resp.status_code}")
                continue

            data = resp.json()
            post_data = data[0]["data"]["children"][0]["data"]

            # Extract fields
            title = post_data.get("title", "").strip()
            author = post_data.get("author", "").strip()
            content = post_data.get("selftext", "").strip()

            # Write to single text file
            outfile.write(f"Post #{i}\n")
            outfile.write(f"Title: {title}\n")
            outfile.write(f"Author: {author}\n")
            outfile.write(f"Content:\n{content if content else '[No text content]'}\n")
            outfile.write("\n" + "=" * 100 + "\n\n")

            print(f"✅ ({i}) Saved: {title[:60]}...")
            time.sleep(2)  # polite delay

        except Exception as e:
            print(f"❌ ({i}) Error scraping {url}: {e}")
            continue

print(f"\n🎉 Done! All posts saved in '{output_file}'")
