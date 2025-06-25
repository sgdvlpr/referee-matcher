import requests
import json
import os
import time

BASE_URL = "https://api.openalex.org"

def fetch_all_openalex_topics(save_dir="./data", filename="openalex_topics.json"):
    base_url = "https://api.openalex.org/topics"
    per_page = 200
    cursor = "*"
    all_topics = []

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    while True:
        url = f"{base_url}?per-page={per_page}&cursor={cursor}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        for topic in results:
            subfield = topic.get("subfield")
            subfield_info = {
                "id": subfield["id"],
                "name": subfield["display_name"]
            } if subfield else None

            all_topics.append({
                "id": topic["id"],
                "name": topic["display_name"],
                "description": topic.get("description", ""),
                "keywords": topic.get("keywords", []),
                "subfield": subfield_info
            })

        print(f"Fetched {len(results)} topics. Total so far: {len(all_topics)}")

        cursor = data["meta"].get("next_cursor")
        if not cursor:
            break

        time.sleep(0.5)  # Be polite with API

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_topics, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_topics)} topics to {save_path}")
    return save_path, len(all_topics)
