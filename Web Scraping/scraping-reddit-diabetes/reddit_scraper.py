import requests
import time
import csv  # 🟢 new import for writing CSV

subreddit = "diabetes"
base_url = f"https://www.reddit.com/r/{subreddit}/.json"
headers = {"User-Agent": "my-scraper/0.1 (by u/yourusername)"}
output_file = f"{subreddit}_posts_and_comments.txt"
csv_file = f"{subreddit}_post_links.csv"  # 🟢 new CSV output file

# 🟢 List to store all post links
post_links = []

# Helper function to recursively extract all comments
def extract_comments(comment_list, level=0):
    comments_data = []
    for c in comment_list:
        kind = c.get("kind")
        if kind != "t1":  # "t1" means a comment
            continue
        data = c.get("data", {})
        body = data.get("body", "[deleted]")
        author = data.get("author", "[deleted]")
        score = data.get("score", 0)
        indent = "  " * level
        comments_data.append(f"{indent}- {author}: {body} (score: {score})\n")
        # Recursively add replies
        replies = data.get("replies")
        if isinstance(replies, dict):
            replies_data = replies.get("data", {}).get("children", [])
            comments_data.extend(extract_comments(replies_data, level + 1))
    return comments_data


# Main scraper
after = None
page = 1
total_posts = 0
total_comments = 0

with open(output_file, "w", encoding="utf-8") as file:
    file.write(f"Reddit Scrape: r/{subreddit}\n")
    file.write("=" * 100 + "\n\n")

    while True:
        params = {"limit": 50}
        if after:
            params["after"] = after

        print(f"📄 Fetching page {page} ...")
        resp = requests.get(base_url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"⚠️ Error: {resp.status_code}")
            break

        data = resp.json()
        posts = data["data"]["children"]

        if not posts:
            break

        for post in posts:
            p = post["data"]
            title = p.get("title")
            author = p.get("author")
            permalink = p.get("permalink")
            post_url = f"https://www.reddit.com{permalink}"  # 🟢 this will be saved to CSV
            post_links.append([title, author, post_url])      # 🟢 save post info
            post_id = p.get("id")
            score = p.get("score")
            created_utc = p.get("created_utc")

            file.write(f"📌 POST: {title}\n")
            file.write(f"Author: {author}\nScore: {score}\nCreated: {created_utc}\nLink: {post_url}\n")
            file.write("-" * 100 + "\n")

            # Fetch comments for this post (optional)
            comments_url = f"https://www.reddit.com{permalink}.json"
            comment_resp = requests.get(comments_url, headers=headers)
            if comment_resp.status_code != 200:
                file.write(f"⚠️ Failed to fetch comments (status {comment_resp.status_code})\n\n")
                continue

            comment_json = comment_resp.json()
            comment_tree = comment_json[1]["data"]["children"]
            comments = extract_comments(comment_tree)

            if comments:
                for c in comments:
                    file.write(c)
                total_comments += len(comments)
            else:
                file.write("No comments found.\n")

            file.write("\n" + "=" * 100 + "\n\n")
            total_posts += 1

            time.sleep(2)

        after = data["data"].get("after")
        if not after:
            break

        page += 1
        time.sleep(3)

# 🟢 After scraping, save all links to CSV
with open(csv_file, "w", newline="", encoding="utf-8") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["Title", "Author", "URL"])
    writer.writerows(post_links)

print(f"✅ Done! Scraped {total_posts} posts and ~{total_comments} comments.")
print(f"🔗 All post links saved to '{csv_file}'.")
