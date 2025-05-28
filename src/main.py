import requests
import time
import os, json

from sentence_transformers import SentenceTransformer, util
import torch

import google.generativeai as genai

from typing import List, Dict

genai.configure(api_key="AIzaSyAj6k2V-Dj39KDMU92nA7q2vYYbsuQ9q2g")
model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-pro"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "primary_topic_works.json")

def load_openalex_topics(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_top_topic_matches_with_st(abs_topic, topics_data, top_n=100):
    model = SentenceTransformer('all-mpnet-base-v2')

    # Embed abstract topic
    abs_emb = model.encode(abs_topic, convert_to_tensor=True)

    candidate_texts = [topic['name'] for topic in topics_data]
    candidate_ids = [topic['id'] for topic in topics_data]
    candidate_descs = [topic.get('description', '') for topic in topics_data]

    # Embed candidate topics
    candidate_embs = model.encode(candidate_texts, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.cos_sim(abs_emb, candidate_embs)[0]

    # Get top N scores and indices
    top_results = cos_scores.topk(top_n)

    top_topics = []
    for score, idx in zip(top_results[0], top_results[1]):
        top_topics.append({
            'id': candidate_ids[idx],
            'name': candidate_texts[idx],
            'description': candidate_descs[idx],
            'score': float(score)
        })

    return top_topics

def find_top_topic_matches_with_gemini(abs_topic: str, candidate_topics: list, n_matches: int = 5):
    """
    Use Gemini to find the best N topic matches for a given abstract topic.
    
    Args:
        abs_topic (str): The topic from the abstract.
        candidate_topics (list): A list of dictionaries with 'id' and 'display_name'.
        model (GenerativeModel): The Gemini model instance.
        top_n (int): Number of top matches to return.

    Returns:
        list of dict: Each dict has 'id', 'name', and 'reason' fields.
    """
    print(type(candidate_topics))
    candidate_names = [topic['name'] for topic in candidate_topics]
    candidate_list = "\n".join(f"- {name}" for name in candidate_names)

    prompt = f"""
        You are an expert research assistant specialized in electrical engineering. Your task is to help identify the most technically 
        and semantically relevant research topics for a given abstract-level research concept.

        Given the abstract-level topic: "{abs_topic}", and the following list of research topics from OpenAlex, rank and select the 
        {n_matches} most relevant matches. Before selecting matches, carefully **read the description of each topic** to understand 
        its research scope and focus.


        Each topic in the candiate_topics is a real entry from the OpenAlex database. You must:
        - Choose matches **only** from this list — do not create, merge, rename, or hallucinate new topics.
        - Use **exact topic names** as they appear in the list.
        
        Your output should consider the following:
        - How technically close the topic is to the concept 
        - Whether the topic is directly involved in research using the given concept
        - Semantic and conceptual relevance
        - Practical applications within electrical engineering

        For each selected match, return:
        - "name": The topic's name (from the list)
        - "id": The topic’s OpenAlex ID (from the list)
        - "reason": A 1–2 sentence explanation why this is a strong match
        - "score": A relevance score from 0 to 1 (1 being the most relevant)

        Return your answer in **JSON format** as a list:
        [
        {{
            "name": "...",
            "id": "...",
            "reason": "...",
            "score": ...
        }},
        ...
        ]

        Here is the list of candidate topics:
        {candidate_list}
    """

    response = model.generate_content(prompt)
    text = response.text
    json_start = text.find("[")
    json_end = text.rfind("]") + 1
    json_str = text[json_start:json_end]

    try:
        matches = json.loads(json_str)

        # Add the corresponding topic ID to each match

        matched_ids = []
        for match in matches:
            topic_obj = next((t for t in candidate_topics if t["name"].lower() == match["name"].lower()), None)
            if topic_obj:
                matched_ids.append(topic_obj["id"])

        # Pretty-print the matches
        print("\n🔍 Top Matched Topics:\n" + "-"*50)
        for i, match in enumerate(matches, 1):
            print(f"#{i}")
            print(f"Name   : {match.get('name', 'N/A')}")
            print(f"Score  : {match.get('score', 'N/A')}")
            print(f"Reason : {match.get('reason', 'N/A')}")
            print(f"ID     : {match.get('id', 'N/A')}")
            print("-" * 50)

        return matched_ids

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return []

def get_works_by_topic(topic_id: str, min_year: int = 2010, min_citations: int = 30, per_page: int = 100, delay: float = 0.5) -> list:
    base_url = "https://api.openalex.org/works"
    cursor = "*"
    all_works = []

    while True:
        filters = [
            f"primary_topic.id:{topic_id}",
            f"publication_year:>{min_year}",
            f"cited_by_count:>{min_citations}"
        ]
        params = {
            "filter": ",".join(filters),
            "per-page": per_page,
            "sort": "cited_by_count:desc",
            "cursor": cursor,
            "mailto": "scramjet14@gmail.com"
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        works = data.get("results", [])
        all_works.extend(works)

        if not data.get("meta", {}).get("next_cursor"):
            break
        cursor = data["meta"]["next_cursor"]
        time.sleep(delay)  # ⏳ Be polite

    
    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_works, f, indent=2)

    print(f"\n📁 Saved {len(all_works)} works to {OUTPUT_FILE}")
    return all_works

def get_keywords_by_works(works, rank=1):
    """
    Extract the N-th ranked primary keyword (by score) from each work.

    Parameters:
    - works: list of OpenAlex work dicts
    - rank: integer from 1 to 5 specifying which ranked keyword to extract

    Returns:
    - A list of distinct keywords at the given rank
    """
    assert 1 <= rank <= 5, "Rank must be between 1 and 5"

    ranked_keywords_set = set()

    for work in works:
        keywords = work.get("keywords", [])
        if not keywords:
            continue

        sorted_kws = sorted(keywords, key=lambda k: k.get("score", 0), reverse=True)

        if len(sorted_kws) >= rank:
            name = sorted_kws[rank - 1].get("display_name")
            if name:
                ranked_keywords_set.add(name)

    ranked_keywords_list = list(ranked_keywords_set)
    print(f"\n✨ Extracted {len(ranked_keywords_list)} distinct rank-{rank} primary keywords.")
    return ranked_keywords_list

def find_top_matches(abstract_keywords, works_keywords, top_k=5):
    model = SentenceTransformer('all-mpnet-base-v2')

    # Embed both lists
    abstract_embeds = model.encode(abstract_keywords, convert_to_tensor=True)
    works_embeds = model.encode(works_keywords, convert_to_tensor=True)

    results = {}

    for i, abs_kw in enumerate(abstract_keywords):
        # Compute cosine similarities for one abstract keyword vs all works keywords
        cosine_scores = util.cos_sim(abstract_embeds[i], works_embeds)[0]

        # Get top_k results indices
        top_results = torch.topk(cosine_scores, k=top_k)

        # Store matched keywords with their scores
        matches = []
        for score, idx in zip(top_results.values, top_results.indices):
            kw = works_keywords[idx]
            matches.append({"keyword": kw, "score": float(score)})

        results[abs_kw] = matches

    return results

def find_top_kw_matches_with_gemini(abstract_keywords, works_keywords, top_k=5):
    """
    Uses Gemini to match each abstract keyword to the most semantically related keywords
    from a list of candidate keywords (e.g., from OpenAlex works).
    """
    prompt = f"""
        You are a highly skilled research assistant in the domain of electrical engineering.
        You are given:

        1. A list of at most 5 **technical keywords** from a research abstract.
        2. A large list of **candidate technical keywords** extracted from related research papers.

        Your task is to find the top {top_k} most **semantically and technically relevant** candidate keywords for **each** abstract keyword.

        Guidelines:
        - Work individually on each abstract keyword.
        - Only use keywords from the given candidate list. Do not create or rephrase keywords.
        - Avoid overly generic terms unless explicitly relevant.
        - Avoid keyword drift into unrelated fields.
        - Prefer domain overlap, shared subfields, or technical hierarchies.
        - Prioritize keywords that would appear in the same paper, review, or methodology.

        Return your answer in **JSON** format as:

        {{
            "matches": [
                {{
                    "abstract_keyword": "abstract keyword",
                    "matches": [
                        {{ "match": "candidate keyword", "reason": "short reason (optional)" }},
                        ...
                    ]
                }},
                ...
            ]
        }}

        Abstract Keywords:
        {abstract_keywords}

        Candidate Keywords:
        {works_keywords}
    """
    response = model.generate_content(prompt)
    try:
        text = response.text
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        json_str = text[json_start:json_end]

        results = json.loads(json_str)
        return results.get("matches", [])
    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Raw response:", response.text)
        return []

def get_authors_for_topic(topic_id, top_n=10):
    # Your existing function to query authors related to the topic ID,
    # sorted by expertise/score.
    pass

def aggregate_authors_by_topic(topics_with_scores):
    """
    topics_with_scores: [
      {"id": "T1234", "name": "...", "score": 0.7},
      {"id": "T2345", "name": "...", "score": 0.2},
      {"id": "T3456", "name": "...", "score": 0.1}
    ]
    """
    n_authors_per_topic = [10, 8, 6]
    final_authors = []
    for i, topic in enumerate(topics_with_scores[:3]):
        authors = get_authors_for_topic(topic['id'], top_n=n_authors_per_topic[i])
        final_authors.extend(authors)

    return final_authors

# Usage

topic_ids = ["T10323"]  

# Usage

topics_data = load_openalex_topics("./data/openalex_topics.json")
abstract_topic = "mm-wave circuit design"

#candidate_list = find_top_topic_matches_with_st(abstract_topic, topics_data, top_n=100)

topic_ids = find_top_topic_matches_with_gemini(abstract_topic, topics_data, n_matches=5)
print(topic_ids)

primary_topic_id = topic_ids[0]

works = get_works_by_topic(primary_topic_id, min_year=2015, min_citations=90)
print(f"\n🔎 Total works fetched: {len(works)}")

works_kws = get_keywords_by_works(works, 1)
print(works_kws)

# # Example usage:
abs_kws = ["Mixer design"]

matches = find_top_kw_matches_with_gemini(abs_kws, works_kws, top_k=2)
#print(json.dumps(matches, indent=2))

for entry in matches:
    abstract_kw = entry.get("abstract_keyword", "Unknown keyword")
    print(f"\n🔹 Matches for: {abstract_kw}")
    for match in entry.get("matches", []):
        print(f"- {match['match']}: {match.get('reason', 'No reason provided')}")

