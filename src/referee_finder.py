import requests, httpx
import asyncio
import google.generativeai as genai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
import re
from typing import List, Dict, Optional
from pathlib import Path

genai.configure(api_key="AIzaSyAj6k2V-Dj39KDMU92nA7q2vYYbsuQ9q2g")
model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-pro"

OPENALEX_API_BASE = "https://api.openalex.org"

class RefereeMatcher:
    def __init__(self, author_ids: list[str]):
        self.AUTHOR_IDS = author_ids
        self.base_url = "https://api.openalex.org"
        self.fields_metadata = Path(__file__).parent / "data" / "openalex_fields_metadata.json" 

    async def set_conflict_ids(self):
        """
        Sets the global CONFLICT_IDS variable using concurrent coauthor fetching.

        It updates:
        - CONFLICT_IDS: a deduplicated list of all coauthor IDs 
        associated with the submitting authors, used to detect conflicts.

        Returns:
            List of deduplicated conflict IDs.
        """

        coauthor_dict = self.get_all_coauthors_concurrently()
        all_ids = {co_id for coauthors in coauthor_dict.values() for co_id in coauthors}
        self.CONFLICT_IDS = list(all_ids)
        print("self_conflicts", self.CONFLICT_IDS)

    async def get_topics_for_subfield(self, field_id: str, subfield_id: str) -> List[Dict]:
        """
        Fetch all topics related to a given subfield from the metadata file.
        Each topic includes id, name, description, and optional keywords.
        """
        
        with open(self.fields_metadata, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        for field in self.metadata:
            if field.get("field_id") == field_id:
                for subfield in field.get("subfields", []):
                    if subfield.get("subf_id") == subfield_id:
                        return subfield.get("subf_topics", [])
                print(f"[WARN] Subfield ID {subfield_id} not found under field {field_id}.")
                return []
        
        print(f"[WARN] Field ID {field_id} not found.")
        return []

    async def get_top_works_for_topic(self, topic_id: str, top_n: int = 20, from_year: int = 2000, min_citations: int = 20):
        """
        Fetch top works from OpenAlex under a given topic ID.
        Returns a list of structured works including author metadata.
        """
        base_url = f"{self.base_url}/works"
        short_id = topic_id.split("/")[-1]
        filters = [
            f"topics.id:{short_id}",
            f"publication_year:>{from_year}",
            f"cited_by_count:>{min_citations}"
        ]

        params = {
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",  # Descending sort by citations
            "per-page": 100,
            "mailto": "scramjet14@gmail.com"
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data for topic {topic_id}. Status code: {response.status_code}")
            return []

        works = response.json().get("results", [])

        top_works = []
        for work in works[:top_n]:
            authorships = work.get("authorships", [])

            if authorships:
                top_authorship = authorships[0]
                top_inst = top_authorship.get("institutions", [])
                top_referee = {
                    "name": top_authorship["author"]["display_name"],
                    "id": top_authorship["author"].get("id"),
                    "institution": top_inst[0]["display_name"] if top_inst else None
                }

                # Extract co-authors
                referees = []
                for referee in authorships[1:]:
                    insts = referee.get("institutions", [])
                    referees.append({
                        "name": referee["author"]["display_name"],
                        "id": referee["author"].get("id"),
                        "institution": insts[0]["display_name"] if insts else None
                    })
            else:
                top_referee = {
                    "name": "Unknown",
                    "id": None,
                    "institution": None
                }
                referees = []

            abstract_index = work.get("abstract_inverted_index", None)
            if abstract_index:
                abstract_text = self.abstract_index_to_text(abstract_index)
            else:
                abstract_text = ""
            top_works.append({
                "title": work.get("display_name"),
                "abstract": abstract_text,
                "top_referee": top_referee,
                "alt_referees": referees,
                "citations_count": work.get("cited_by_count"),
                "year": work.get("publication_year"),
                "url": work.get("id")
            })

        return top_works
  
    # Just for illustration - build on top of that when you are back on 1404/03/17
    async def extract_topics_from_abstract(self, abstract: str, subfield_topics: List[Dict]) -> List[Dict]:
        """
        Uses Gemini to select the most relevant topics from subfield_topics based on abstract.
        Returns a list of matching topic dicts.
        """
        # Build prompt for Gemini
        topic_descriptions = "\n".join(
            f"{idx+1}. {t['topic_name']} - Keywords: {', '.join(t.get('keywords', []))}\nDescription: {t.get('description', 'No description')}"
            for idx, t in enumerate(subfield_topics)
        )
        
        prompt = (
            f"Given the abstract of a research paper:\n\"\"\"\n{abstract}\n\"\"\"\n\n"
            f"Select the most relevant topics from the following list (by number) that best match this abstract. "
            f"Return the numbers as a comma-separated list.\n\nTopics:\n{topic_descriptions}\n"
        )

        # Call Gemini LLM (replace with your actual async API call)
        response = await self.call_gemini(prompt)

        # Parse response to get topic numbers
        # Expected output example: "1, 3, 5"
        selected_indices = []
        try:
            # Extract numbers from response
            selected_indices = [int(num.strip()) - 1 for num in response.split(",") if num.strip().isdigit()]
        except Exception:
            # Fallback empty if parsing fails
            selected_indices = []

        # Filter topics based on indices
        matched_topics = [subfield_topics[i] for i in selected_indices if 0 <= i < len(subfield_topics)]

        return matched_topics
    
    # Just for illustration - build on top of that when you are back on 1404/03/17
    async def filter_relevant_works(self, abstract: str, works: List[Dict]) -> List[Dict]:
        """
        Use Gemini to evaluate and filter works that are highly relevant to the abstract.
        Returns a filtered list of works.
        """
        prompt = f"""
        You are helping select scientific works that are highly relevant to a given abstract.

        Abstract:
        \"\"\"
        {abstract}
        \"\"\"

        Below is a list of candidate works. Each has a title and short description. Which ones are most relevant to the abstract?

        Candidate works:
        {json.dumps([
            {"title": w["title"], "venue": w.get("venue"), "year": w.get("year")}
            for w in works
        ], indent=2)}

        Return the list of most relevant work titles as a JSON array.
        """

        response = self.llm_chat(prompt)
        selected_titles = self._parse_json_list_response(response)

        matched_works = [w for w in works if w["title"] in selected_titles]
        return matched_works

    async def get_top_referees(self, works: list[dict]) -> list[dict]:
        """
        Extract all unique top referees from a list of works.
        Each referee includes name, OpenAlex ID, and institution.
        Submitting authors are excluded but their co-authors are flagged.
        """
        await self.set_conflict_ids()

        top_referees = []
        seen_ids = set()

        for work in works:
            referee = work.get("top_referee", {})
            referee_id = referee.get("id")

            # Exclude if referee is a submitting author or already seen or conflict
            if (
                referee_id
                and referee_id not in seen_ids
                and referee_id not in self.AUTHOR_IDS
            ):
                seen_ids.add(referee_id)
                top_referees.append({
                    "name": referee.get("name"),
                    "id": referee_id,
                    "institution": referee.get("institution"),
                    "score": 0.0,
                    "is_conflict": self.is_conflict(referee_id)
                })

        return top_referees

    def get_coauthors(self, author_id: str, max_pubs=500) -> tuple[str, dict]:
        """
        Fetch all publications for the submitting author and return a dictionary of co-author details.
        Used for conflict detection.
        Returns:
            (author_id, coauthors) where:
                author_id (str): The submitting author's ID.
                coauthors (dict): Maps co-author ID to a dictionary with keys:
                    - 'name': co-author's name
                    - 'institution': co-author's primary institution (if available)
        """
        base_url = "https://api.openalex.org/works"
        filters = [f"author.id:{author_id}"]
        params = {
            "filter": ",".join(filters),
            "per-page": 200,
            "mailto": "scramjet14@gmail.com"
        }

        coauthors = {}
        cursor = "*"
        total_pubs = 0

        while cursor and total_pubs < max_pubs:
            params["cursor"] = cursor
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching publications for author {author_id}: {response.status_code}")
                break
            
            data = response.json()
            works = data.get("results", [])
            total_pubs += len(works)

            for work in works:
                for authorship in work.get("authorships", []):
                    co_author = authorship.get("author", {})
                    co_author_id = co_author.get("id")
                    if co_author_id and co_author_id != author_id:
                        institution = None
                        # Extract institution if available
                        institution_info = authorship.get("institutions", [])
                        if institution_info:
                            institution = institution_info[0].get("display_name")  # take first institution's name
                        
                        # Store info keyed by co-author ID
                        coauthors[co_author_id] = {
                            "name": co_author.get("display_name"),
                            "institution": institution
                        }
            
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        
        return author_id, coauthors

    def get_all_coauthors_concurrently(self, max_pubs=500) -> dict:
        """
        Fetch co-author details concurrently for multiple authors, 
        ensuring none of the coauthors are the submitting authors themselves.
        Used for conflict detection.

        Args:
            author_ids (list[str]): List of author IDs to fetch coauthors for.
            submitting_author_ids (set or list[str]): IDs of submitting authors to exclude from coauthors.
            max_pubs (int, optional): Maximum number of publications to fetch per author. Defaults to 500.

        Returns:
            dict: A dictionary mapping each author ID to a dictionary of their coauthors.
                Each coauthor dictionary maps coauthor ID to their details (e.g., name, institution).
                Coauthors who are submitting authors themselves are excluded.
        """
        
        results = {}

        def get_filtered_coauthors(author_id):
            author_id, coauthors = self.get_coauthors(author_id, max_pubs)
            # Filter out submitting authors from coauthors
            filtered_coauthors = {}
            for co_id, info in coauthors.items():
                if co_id not in self.AUTHOR_IDS:
                    filtered_coauthors[co_id] = info

            return author_id, filtered_coauthors

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_filtered_coauthors, author_id) for author_id in self.AUTHOR_IDS]

            for future in as_completed(futures):
                author_id, coauthors = future.result()
                results[author_id] = coauthors

        return results

    def is_conflict(self, candidate_referee_id: str) -> bool:
        return candidate_referee_id in self.CONFLICT_IDS
 
    async def get_author_details(self, author_id):
        author_id = author_id.rsplit("/", 1)[-1]
        url = f"{OPENALEX_API_BASE}/authors/{author_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_author_works(self, author_id, max_works=200):
        url = f"{OPENALEX_API_BASE}/works"
        params = {
            "filter": f"author.id:{author_id}",
            "per-page": max_works,
            "sort": "cited_by_count:desc"
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            works = []
            for work in data.get("results", []):
                abstract_index = work.get("abstract_inverted_index", None)
                if abstract_index:
                    abstract_text = self.abstract_index_to_text(abstract_index)
                else:
                    abstract_text = ""
                works.append({
                    "id": work["id"],
                    "title": work.get("title", ""),
                    "abstract": abstract_text,
                    "publication_year": work.get("publication_year", None),
                    "citation_count": work.get("cited_by_count", 0),
                })
            return works

    async def get_referee_details(self, author_id: str, max_works=200):
        # Run both requests concurrently
        author_data, works = await asyncio.gather(
            self.get_author_details(author_id),
            self.get_author_works(author_id, max_works=max_works),
        )

        #Get topic share dict for easy lookup
        topic_share_map = {
            topic["id"]: topic["value"]
            for topic in author_data.get("topic_share", [])
        }

        topics = []
        for topic in author_data.get("topics", []):
            tid = topic['id']
            topics.append({
                "id": tid,
                "name": topic["display_name"],
                "count": topic.get("count", 0),
                "topic_share": topic_share_map.get(tid, 0.0),
            })

        last_known_institutions = author_data.get("last_known_institutions", [])
        if last_known_institutions:
            last_known_institution_name = last_known_institutions[0].get("display_name")
        else:
            last_known_institution_name = None

        response = {
            "author_id": author_data["id"],
            "name": author_data.get("display_name", ""),
            "summary_stats": author_data.get("summary_stats", {}),
            "last_known_institution": last_known_institution_name,
            "topics": topics,
            "works": works,
        }
        return response
    
    async def get_all_referees_details(self, author_ids: list[str], max_works=200):
        # Create a list of coroutines for each referee detail request
        coroutines = [self.get_referee_details(author_id, max_works=max_works) for author_id in author_ids]
        
        # Run them concurrently and wait for all to finish
        all_details = await asyncio.gather(*coroutines)
        
        return all_details

    def abstract_index_to_text(self, abstract_index: dict) -> str:
        """
        Convert a reversed abstract index (word -> list of positions)
        back to a plain abstract text string, ordered by positions,
        cleaned of special characters like \r and \n.

        Parameters:
            abstract_index (dict): keys are words, values are lists of int positions.

        Returns:
            str: reconstructed plain abstract text.
        """
        # Create a map position -> word for all positions
        pos_word_map = {}
        for word, positions in abstract_index.items():
            for pos in positions:
                pos_word_map[pos] = word

        # Sort positions to get words in order
        sorted_positions = sorted(pos_word_map.keys())

        # Join words by their sorted position
        words_in_order = [pos_word_map[pos] for pos in sorted_positions]
        reconstructed = ' '.join(words_in_order)

        # Clean special characters \r, \n, multiple spaces
        cleaned = re.sub(r'[\r\n]+', ' ', reconstructed)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned
        
async def main():

    candidate_author_ids = [
        "https://openalex.org/authors/A5050993786", 
        "https://openalex.org/A5109035399",
        "https://openalex.org/authors/A5001986671",
        "https://openalex.org/authors/A5036198731",
        "https://openalex.org/authors/A5110066788",
        ]

    submitting_authors_ids = [
        'https://openalex.org/A5088589128', 
        'https://openalex.org/A5072058881', 
        'https://openalex.org/A5041807782',
        'https://openalex.org/A5087380109'
    ]

    matcher = RefereeMatcher(submitting_authors_ids)

    field_id = "https://openalex.org/fields/27"
    subf_id = "https://openalex.org/subfields/2703"
    topics = await matcher.get_topics_for_subfield(field_id, subf_id)
    pprint(topics, compact=False, width=100)

    # referee_id = 'https://openalex.org/A5053004808'
    
    # paper_title = "A 32-mW 40-Gb/s CMOS NRZ Receiver"

    # topic_id ="https://openalex.org/T14117"

    # top_works = await matcher.fetch_top_works_for_topic(topic_id, 10, 2010, 50)
    # pprint(top_works, compact=False, width=100)

    # works = await matcher.get_top_works_by_topic(paper_title, 10)

    # pprint(works, compact=False, width=100)

    # # Get detailed referee information
    # details = await matcher.get_all_referees_details(candidate_author_ids)

    # pprint(details, compact=False, width=100)

# Run main
asyncio.run(main())