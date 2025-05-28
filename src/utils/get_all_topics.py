import requests
import json
import os
import time
import pprint

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

def fetch_paginated_data(url):
    per_page = 200
    cursor = "*"
    results = []
    page = 1

    while cursor:
        full_url = f"{url}?per-page={per_page}&cursor={cursor}"
        response = requests.get(full_url)
        if response.status_code != 200:
            print(f"❌ Failed to fetch from {full_url}: {response.status_code}")
            break

        data = response.json()
        page_results = data.get("results", [])
        results.extend(page_results)
        cursor = data.get("meta", {}).get("next_cursor")

        print(f"📄 Page {page}: Fetched {len(page_results)} items, Total so far: {len(results)}")
        page += 1
        time.sleep(0.5)  # To avoid overwhelming the API

    return results

def fetch_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to fetch from {url}: {response.status_code}")
        return None

def fetch_paginated_data(url):
    per_page = 200
    cursor = "*"
    results = []

    while cursor:
        full_url = f"{url}?per-page={per_page}&cursor={cursor}"
        response = requests.get(full_url)
        if response.status_code != 200:
            print(f"Failed to fetch from {full_url}: {response.status_code}")
            break
        data = response.json()
        results.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(0.5)  # Be polite
    return results

def build_fields_subfields_topics(save_path="./data/fields_subfields_topics.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fields_url = f"{BASE_URL}/fields"
    all_fields_raw = fetch_paginated_data(fields_url)

    final_data = []

    for field in all_fields_raw:
        field_entry = {
            "field_name": field["display_name"],
            "field_descriptiopn": field["description"],
            "field_alt_names": field["display_name_alternatives"],
            "field_id": field["id"],
            "subfields": []
        }

        for subfield in field.get("subfields", []):
            subf_id = subfield["id"]
            subf_data = {
                "subf_name": subfield["display_name"],
                "subf_id": subfield["id"],
                "subf_topics": []
            }

            # Fetch topics for this subfield
            topics = fetch_topics_from_subfield(subf_id)
            for topic in topics:
                subf_data["subf_topics"].append({
                    "topic_name": topic["display_name"],
                    "topic_id": topic["id"]
                })

            field_entry["subfields"].append(subf_data)

        final_data.append(field_entry)
        print(f"✅ Completed field: {field['display_name']}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All data saved to {save_path}")
    return save_path

def fetch_topics_from_subfield(subfield_id):
    """Fetch topics directly from the subfield JSON object"""
    subfield_short_id = subfield_id.split("/")[-1]
    url = f"{BASE_URL}/subfields/{subfield_short_id}"
    subfield_data = fetch_json(url)
    
    if not subfield_data:
        print(f"❌ Failed to fetch subfield data for {subfield_id}")
        return []

    topics = subfield_data.get("topics", [])
    print(f"✅ Found {len(topics)} topics in subfield '{subfield_data['display_name']}'")
    return topics

def load_topic_keywords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        topics = json.load(f)
    return {
        topic["id"]: topic.get("keywords", [])
        for topic in topics
    }

def load_topic_description(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        topics = json.load(f)
    return {
        topic["id"]: topic.get("description", "")
        for topic in topics
    }    

def enrich_fields_with_description(fields_path, description_path, output_path=None):
    # Load data
    with open(fields_path, "r", encoding="utf-8") as f:
        fields_data = json.load(f)

    topic_description_map = load_topic_description(description_path)

    # Add keywords to each topic
    for field in fields_data:
        for subfield in field.get("subfields", []):
            for topic in subfield.get("subf_topics", []):
                topic_id = topic["topic_id"]
                if topic_id in topic_description_map:
                    topic["description"] = topic_description_map[topic_id]

    # Save enriched file
    if not output_path:
        output_path = fields_path  # overwrite if no output specified

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fields_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Descriptions added to topics and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    enrich_fields_with_description(
        fields_path="./data/fields_subfields_topics_enriched.json",
        description_path="./data/openalex_topics.json",
        output_path="./data/fields_subfields_topics_enriched_v1.json"  # or None to overwrite
    )
