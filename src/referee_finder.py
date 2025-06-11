import requests, httpx
import asyncio
import google.generativeai as genai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
import re
from typing import List, Dict, Optional
from pathlib import Path
from asyncio import Semaphore

LIARA_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySUQiOiI2ODQ0NGNjYmMyODAzYzlkYmI0ZTQ3MTIiLCJ0eXBlIjoiYXV0aCIsImlhdCI6MTc0OTMwNzAwMH0.4xuTmA2p0onfvGHo8XlwV4vIjlcE7REMVre0luU-2yU'

from openai import OpenAI

genai.configure(api_key="AIzaSyAj6k2V-Dj39KDMU92nA7q2vYYbsuQ9q2g")
model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-pro"

OPENALEX_API_BASE = "https://api.openalex.org"

class RefereeMatcher:
    def __init__(self, author_ids: list[str]):
        self.AUTHOR_IDS = author_ids
        self.base_url = "https://api.openalex.org"
        self.fields_metadata = Path(__file__).parent / "data" / "openalex_fields_metadata.json" 
        self.subfields = Path(__file__).parent / "data" / "subfields.json" 
        self.llm_client = OpenAI(
            base_url="https://ai.liara.ir/api/v1/68444daf64f28c83a27063e1",
            api_key=LIARA_API_KEY
        )
        self.mailto = "scramjet14@gmail.com"

    async def query_llm(self, prompt):
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

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
  
    def extract_json_array(self, text: str) -> str:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text  # fallback to original if no brackets found

    async def get_best_matching_fields(self, abstract: str) -> Dict:
        """
        Use LLM to find the most relevant subfields from OpenAlex metadata based on the abstract.
        Returns a list of dicts with subfield name, ID, and reason.
        """
        with open(self.subfields, "r", encoding="utf-8") as f:
            subfields_data = json.load(f)

        # Minify for prompt
        subfield_names = [entry["subf_name"] for entry in subfields_data]

        prompt = f"""
        You are a research field classification expert. Your task is to analyze scientific abstracts and determine the **most appropriate academic subfields** for each one.

        You will be given:
        - A list of candidate subfield names
        - A research abstract

        Your goal is to select the **three most relevant subfields** based on the abstract’s core contribution, scientific context, and application area.

        Respond only with a JSON array of three objects in this format:
        {{
            "selected_subfield_name": "Subfield Name",
            "reason": "A short explanation for why this subfield is the best match."
        }}

        Abstract:
        \"\"\"
        {abstract}
        \"\"\"

        Candidate subfields:
        {json.dumps(subfield_names, indent=2)}
        """

        answer = await self.query_llm(prompt)
        clean_answer = self.extract_json_array(answer)

        try:
            selected = json.loads(clean_answer)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("Cleaned text was:", repr(clean_answer))
            return []

        # Add back the subfield IDs from original file
        name_to_id = {entry["subf_name"]: entry["subf_id"] for entry in subfields_data}
        enriched = []
        for item in selected:
            name = item.get("selected_subfield_name")
            enriched.append({
                "selected_subfield_name": name,
                "selected_subfield_id": name_to_id.get(name, ""),
                "reason": item.get("reason", "")
            })

        return enriched

    async def get_topics_for_selected_subfields(self, top_subfields: list) -> Dict[str, List[Dict[str, str]]]:
        """
        Given a JSON string with selected subfields, find and return their topics from the OpenAlex metadata file.
        """
        # Load the OpenAlex metadata from file
        with open(self.fields_metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Create output dict
        result = {}

        # Traverse each selected subfield
        for subfield in top_subfields:
            subfield_id = subfield["selected_subfield_id"]
            subfield_name = subfield["selected_subfield_name"]

            # Search the subfield in the metadata
            for field in metadata:
                for sf in field["subfields"]:
                    if sf["subf_id"] == subfield_id:
                        # Extract and format topics
                        topics = [
                            {
                                "topic_name": topic["topic_name"],
                                "topic_id": topic["topic_id"]
                            }
                            for topic in sf.get("subf_topics", [])
                        ]
                        result[subfield_name] = topics
                        break  # subfield found, no need to continue inner loop

        return result
    
    async def filter_relevant_topics_for_subfields(self, abstract: str, subfield_topic_data: List[Dict]) -> List[Dict]:
        """
        Use LLM to filter and return the most relevant topic(s) for each selected subfield,
        based on their match with the abstract.
        
        Parameters:
            abstract (str): The research abstract.
            subfield_topic_data (List[Dict]): Output from `get_topics_for_selected_subfields`, 
                a list where each item has:
                    - subfield_name
                    - subfield_id
                    - topics: List[Dict] with topic_name, topic_id, keywords, description

        Returns:
            List[Dict]: Each item includes:
                - subfield_name
                - subfield_id
                - selected_topic(s) with name, id, reason
        """
        prompt = f"""
            You are an expert in academic field classification. You will be given:
            1. A scientific abstract,
            2. A list of subfields, each with its associated OpenAlex topics (name, keywords, description).

            Your job is to identify the **most relevant topic(s)** (at least 3) under each subfield. You must analyze how closely each topic aligns with the abstract’s core focus, contribution, and terminology.

            Avoid choosing topics just based on keyword overlap — prioritize **semantic alignment and intent**. Return your answer in this exact JSON format:

            [
            {{
                "subfield_name": "Subfield Name",
                "subfield_id": "Subfield ID",
                "selected_topics": [
                {{
                    "topic_name": "Topic Name",
                    "topic_id": "Topic ID",
                    "reason": "Short reason explaining why this topic is a good match for the abstract."
                }}
                ]
            }},
            ...
            ]

            Abstract:
            \"\"\"
            {abstract}
            \"\"\"

            Subfield and topic candidates:
            {subfield_topic_data}
        """

        raw_output = await self.query_llm(prompt)

        # Try to extract the JSON content from inside code block if present
        json_data = self.extract_json_array(raw_output)

        try:
            filtered_topics = json.loads(json_data)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("Raw LLM output:\n", raw_output)
            return []

        return filtered_topics

    def extract_topic_ids(self, topic_data: List[Dict]) -> List[Dict[str, str]]:
        """
        Flattens the nested topic_data structure into a list of dicts
        containing topic_id and topic_name.
        """
        extracted_topics = []
        for subfield in topic_data:
            for topic in subfield.get("selected_topics", []):
                extracted_topics.append({
                    "topic_id": topic["topic_id"],
                    "topic_name": topic["topic_name"]
                })
        return extracted_topics

    async def get_top_works_for_topic(self, topic_id: str, top_n: int = 5, from_year: int = 2016, min_citations: int = 20):
        """
        Async version: Fetch top works under a given topic ID.
        """
        base_url = f"{self.base_url}/works"
        short_id = topic_id.split("/")[-1]
        filters = [
            f"primary_topic.id:{short_id}",
            f"publication_year:>{from_year}",
            f"cited_by_count:>{min_citations}"
        ]

        params = {
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",
            "per-page": 100,
            "mailto": "scramjet14@gmail.com"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(base_url, params=params)
                response.raise_for_status()
            except Exception as e:
                print(f"Error fetching works for topic {topic_id}: {e}")
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
                    referees = [
                        {
                            "name": co["author"]["display_name"],
                            "id": co["author"].get("id"),
                            "institution": co.get("institutions", [{}])[0].get("display_name")
                        }
                        for co in authorships[1:]
                    ]
                else:
                    top_referee = {"name": "Unknown", "id": None, "institution": None}
                    referees = []

                abstract_index = work.get("abstract_inverted_index", None)
                abstract_text = self.abstract_index_to_text(abstract_index) if abstract_index else ""

                top_works.append({
                    "title": work.get("display_name"),
                    "abstract": abstract_text,
                    "top_referee": top_referee,
                    "alt_referees": referees,
                    "citations_count": work.get("cited_by_count"),
                    "year": work.get("publication_year"),
                    "url": work.get("id"),
                    "citation_percentile": work.get("cited_by_percentile_year", {}).get("max", 0)  # Normalized metric
                })

            return top_works

    # Should return unique works - currently it may not
    async def fetch_all_topic_works(client, topics_data, top_n_works=5):
        """
        Concurrently fetch top works for a list of topics and return a flat list.
        """
        tasks = []
        for topic in topics_data[:3]:  # Top 3 only
            topic_id = topic["topic_id"]
            tasks.append(
                asyncio.create_task(client.get_top_works_for_topic(topic_id, top_n=top_n_works))
            )

        all_works_nested = await asyncio.gather(*tasks)

        # Flatten the list of lists into a single list
        flat_works = [work for sublist in all_works_nested for work in sublist]

        print("fetched all topics works")
        return flat_works

    async def sort_works_by_relevance(self, works: list[dict], abstract: str) -> list[dict]:
        """
        Scores a list of works based on their relevance to the given abstract using an LLM.
        Sends only minimal fields: abstract (or title if abstract is missing) + URL for mapping.
        Reattaches full original metadata after scoring.
        """
        # Step 1: Build lookup table and minimal input
        id_to_work = {work["url"]: work for work in works}
        minimal_works = []
        for work in works:
            minimal_works.append({
                "url": work["url"],
                "abstract": work.get("abstract", ""),
                "title": work.get("title", "")
            })

        # Step 2: Prompt
        prompt = f"""
        You are an expert research assistant.

        You are given the abstract of a submitted paper. Your task is to score the **relevance** of a list of prior works to this paper. Relevance should be scored on a scale from **0 (not relevant)** to **10 (highly relevant)**.

        For each work, you are given:
        - Abstract (may be empty)
        - Title 
        - URL (for identification)

        ### Abstract of the submitted paper:
        \"\"\"{abstract}\"\"\"

        ### Instructions:
        1. Use the abstract if present, otherwise use the title.
        2. Score each work with a relevance_score from 0.0 to 10.0 (e.g., 7.3, 9.6).
        3. Keep the original fields (abstract, title, url) and **add** a `relevance_score`.

        Return a JSON array, **sorted from most to least relevant**.

        ### Works to Score:
        {json.dumps(minimal_works, indent=2)}
        """

        raw_output = await self.query_llm(prompt)
        response = self.extract_json_array(raw_output)
        scored_minimal = json.loads(response)

        # Step 3: Reattach full metadata with scores
        enriched = []
        for scored in scored_minimal:
            original = id_to_work.get(scored["url"], {})
            enriched.append({**original, "relevance_score": scored.get("relevance_score", 0)})

        print("Sorted works by relevance")
        return enriched

    async def fetch_recent_referee_works(
        self,
        referee_id: str,
        from_year: int = None,
        min_citations: int = 0,
        max_citations: int = 1_000_000
    ) -> list[dict]:
        """
        Fetch all works authored by `author_id` from `from_year` onward, applying citation filters.
        Continues paginating until no more qualifying works are found or older than `from_year`.
        """
        url = f"{self.base_url}/works"
        cursor = "*"
        all_recent_works = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                params = {
                    "filter": f"author.id:{referee_id}",
                    "sort": "publication_year:desc,cited_by_count:desc",
                    "per-page": 50,
                    "cursor": cursor,
                    "mailto": self.mailto
                }
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                new_filtered = []
                for w in data.get("results", []):
                    year = w.get("publication_year")
                    citations = w.get("cited_by_count", 0)

                    # Stop condition: year too old
                    if from_year and year and year < from_year:
                        return all_recent_works

                    if citations < min_citations or citations > max_citations:
                        continue

                    abstract_index = w.get("abstract_inverted_index")
                    abstract_text = self.abstract_index_to_text(abstract_index) if abstract_index else ""

                    new_filtered.append({
                        "id": w["id"],
                        "title": w.get("title", ""),
                        "abstract": abstract_text,
                        "publication_year": year,
                        "citation_count": citations,
                        "counts_by_year": w.get("counts_by_year", [])
                    })

                all_recent_works.extend(new_filtered)

                next_cursor = data.get("meta", {}).get("next_cursor")
                if not next_cursor or not data.get("results"):
                    break
                cursor = next_cursor

        return all_recent_works

    async def get_top_referees(
        self,
        works: list[dict],
        from_year: int = None,
        min_citations: int = 0,
        max_citations: int = None,
        concurrency_limit: int = 5,  # You can adjust this based on reliability
    ) -> list[dict]:
        """
        Extract unique top referees from a list of works, along with their recent works.
        Submitting authors are excluded. Co-authors are flagged.
        """
        await self.set_conflict_ids()

        seen_ids = set()
        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit)

        for work in works:
            referee = work.get("top_referee", {})
            referee_id = referee.get("id")

            if (
                referee_id
                and referee_id not in seen_ids
                and referee_id not in self.AUTHOR_IDS
            ):
                seen_ids.add(referee_id)

                async def fetch_with_limit(ref=referee):
                    async with semaphore:
                        ref_id = ref["id"]
                        recent_works = await self.fetch_recent_referee_works(
                            referee_id=ref_id,
                            from_year=from_year,
                            min_citations=min_citations,
                            max_citations=max_citations,
                        )
                        return {
                            "name": ref.get("name"),
                            "id": ref_id,
                            "institution": ref.get("institution"),
                            "score": 0.0,
                            "is_conflict": self.is_conflict(ref_id),
                            "works": recent_works,
                        }

                tasks.append(fetch_with_limit())

        top_referees = await asyncio.gather(*tasks)
        print("Top referees received")
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

        # Filter out submitting authors from coauthors
        def get_filtered_coauthors(author_id):
            author_id, coauthors = self.get_coauthors(author_id, max_pubs)
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
    
    def extract_batched_works(self, top_referees: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Extracts and organizes recent works from a list of top referees into a batched dictionary format.

        This function prepares the input needed for AI-assisted work relevance scoring by 
        creating a mapping from referee IDs to a list of their associated works, 
        where each work is represented by its ID, title, and abstract.

        Args:
            top_referees (List[Dict]): A list of dictionaries, where each dictionary represents a referee and contains:
                - 'id' (str): The unique OpenAlex ID of the referee.
                - 'works' (List[Dict]): A list of works authored by the referee, where each work dictionary may include:
                    - 'id' (str): The OpenAlex ID of the work.
                    - 'title' (str): The title of the work.
                    - 'abstract' (str): The abstract of the work.

        Returns:
            List[Dict]: A list of dictionaries, each with 'referee_id' and 'works' fields.
        """
        batched_works = []
        for referee in top_referees:
            referee_id = referee['id']
            works = referee.get("works", [])
            work_list = [
                {
                    "id": work["id"],
                    "title": work.get("title", ""),
                    "abstract": work.get("abstract", "")
                }
                for work in works
            ]
            batched_works.append({
                "referee_id": referee_id,
                "works": work_list
            })
        return batched_works
    
    async def reject_irrelevant_works_from_referee(self, referee: dict, abstract: str, threshold: float = 6.5) -> dict:
        """
        Uses an AI model to evaluate and score the relevance of a referee's works to a given abstract.
        Filters and returns the works that score below a specified relevance threshold.

        Args:
            referee (dict): A dictionary containing:
            - referee_id (str): The OpenAlex ID of the referee.
            - works (list of dict): List of works, each with 'id', 'title', and 'abstract' fields.
            abstract (str): The abstract of the submitted paper to compare against.
            threshold (float, optional): Relevance score threshold. Defaults to 6.5.

        Returns:
            dict: A dictionary with referee_id as key and list of rejected works as values.
                Each rejected work is a dict with 'id', 'relevance_score', and 'reason'.
        """

        # Compose prompt for the AI
        prompt = f"""
            You are an expert research assistant.

            You are given the abstract of a submitted paper. Your task is to evaluate the **relevance** of a list of prior works authored by a candidate referee.

            Each work has a title and an optional abstract, plus an id for identification. For each work:
            - Use the abstract if present, otherwise use the title
            - Provide a `relevance_score` between 0.0 and 10.0 (e.g., 6.7, 9.3)
            - Provide a one-sentence `reason` explaining why the score was assigned
            - Keep the original fields (abstract, title, id) and **add** a `relevance_score`.

            Return a JSON array of objects with keys: id, relevance_score, reason.

            ### Abstract of the submitted paper:
            \"\"\"{abstract}\"\"\"

            ### Works to Evaluate:
            {json.dumps(referee["works"], indent=2)}
        """.strip()

        raw_output = await self.query_llm(prompt)
        response = self.extract_json_array(raw_output)
        scored_works = json.loads(response)

        # Filter works below threshold
        rejected = []
        for scored in scored_works:
            score = scored.get("relevance_score", 0.0)
            if score < threshold:
                rejected.append({
                    "id": scored["id"],
                    "relevance_score": score,
                    "reason": scored.get("reason", "No reason provided")
                })

        print("Filtered batch works by relevance done")
        return {referee["referee_id"]: rejected}

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
    
    abstract0 = 'Injection locking characteristics of oscillators are derived and a graphical analysis is presented that describes injection pulling in time and frequency domains. An identity obtained from phase and envelope equations is used to express the requisite oscillator nonlinearity and interpret phase noise reduction. The behavior of phase-locked oscillators under injection pulling is also formulated.'
    abstract1 = 'This study introduces a deep convolutional neural network model trained on retinal fundus images to automatically detect signs of diabetic retinopathy. Using a dataset of 35,000 annotated images, our model achieves an AUC of 0.96, outperforming existing computer-aided diagnostic tools. The approach demonstrates potential for large-scale automated screening.'
    abstract2 = 'A class-E RF power amplifier operating at 868 MHz is designed and implemented using a 65nm CMOS process for low-power IoT applications. The design achieves a power-added efficiency of 62% with a 10 dBm output power. A compact impedance-matching network and envelope shaping are employed to reduce power consumption.'
    abstract3 = 'We present a hybrid analog-digital beamforming architecture for mmWave massive MIMO systems using lens arrays and sparse channel estimation. Simulation results at 28 GHz show that our approach significantly reduces hardware complexity while achieving spectral efficiencies close to fully digital systems, making it ideal for 5G base stations.'
    abstract4 = 'We propose a transformer-based architecture for few-shot learning that leverages cross-attention and meta-representation adaptation. Our model outperforms existing benchmarks on the MiniImageNet and CIFAR-FS datasets with a 12% increase in classification accuracy, showing its effectiveness in low-data environments.'
    abstract5 = 'A computational study of unsteady flow around a rotating cylinder is conducted using Reynolds-Averaged Navier-Stokes equations. Results indicate a critical transition in vortex shedding frequency at specific Reynolds numbers, providing insight into drag reduction mechanisms for marine applications.'
    abstract6 = 'We investigate the quantum Hall effect in monolayer MoS₂ using low-temperature magnetotransport measurements. Observed plateaus at integer filling factors confirm the presence of a two-dimensional electron gas and support the spin-valley locking hypothesis in transition metal dichalcogenides.'
    abstract7 = 'A retrospective study of 1,200 breast cancer patients revealed that HER2 overexpression is correlated with poorer response to neoadjuvant chemotherapy. The use of trastuzumab improved disease-free survival by 35%, suggesting its continued utility in personalized oncology treatment protocols.'
    abstract8 = 'This study evaluates the seismic performance of reinforced concrete shear walls retrofitted with fiber-reinforced polymers (FRP). Using shake table tests, results show a 40% improvement in ductility and a significant delay in structural failure under simulated earthquake loading.'
    abstract9 = 'Through CRISPR/Cas9 editing, we knocked out the FOXP2 gene in mouse models to investigate its role in vocalization. The edited mice displayed altered ultrasonic vocal patterns, supporting the hypothesis that FOXP2 is critical for speech evolution and neurodevelopment.'
    abstract10 = 'Using satellite-based aerosol optical depth data, we show that particulate emissions in South Asia significantly contribute to regional monsoon suppression. Climate models incorporating this data predict a 12% reduction in seasonal rainfall by 2050 under current emission trajectories.'
    abstract11 = 'This paper explores the impact of framing effects on financial risk-taking among millennials. In a randomized experiment, subjects exposed to gain-framed messages were 24% more likely to invest in high-risk assets, highlighting the significance of behavioral nudges in policy design.'
    abstract12 = 'We report the synthesis of a NiFe-layered double hydroxide nanosheet catalyst for oxygen evolution in alkaline electrolyzers. The catalyst shows an overpotential of only 240 mV at 10 mA/cm² and maintains stability over 100 hours, marking a step toward efficient water splitting.'
    abstract13 = "Quantum many-body systems lie at the heart of modern condensed matter physics, quantum information science, and statistical mechanics. These systems consist of large ensembles of interacting particles whose collective quantum behavior gives rise to rich and often non-intuitive phenomena, such as quantum phase transitions, entanglement, and topological order. Understanding and simulating such systems remains a grand challenge due to the exponential complexity of their Hilbert space. Recent advances, including tensor network methods, quantum Monte Carlo, and machine learning-inspired approaches, have enabled significant progress in capturing the low-energy physics of various models. Moreover, experimental breakthroughs using ultracold atoms, superconducting qubits, and Rydberg atom arrays now allow precise control and observation of many-body dynamics in regimes once thought inaccessible. These developments are paving the way toward unraveling fundamental aspects of quantum matter and advancing technologies such as quantum simulation and computation."

    # field = await matcher.get_best_matching_fields(abstract6)
    # subfields_topics = await matcher.get_topics_for_selected_subfields(field)
    # filters = await matcher.filter_relevant_topics_for_subfields(abstract6, subfields_topics)
    # topic_ids = matcher.extract_topic_ids(filters)
    # top_works = await matcher.fetch_top_works_for_topics(topic_ids)

    topic_id = "https://openalex.org/T10321"

    topics_data = [
        {"topic_id": "https://openalex.org/T10622", "topic_name": "Quantum Mechanics and Applications"},
        {"topic_id": "https://openalex.org/T11804", "topic_name": "Quantum many-body systems"},
        {"topic_id": "https://openalex.org/T10275", "topic_name": "2D Materials and Applications"},
        # ... possibly more, but we want just the top 3
    ]
    
    all_top_works = await matcher.fetch_all_topic_works(topics_data)

    sorted_works_by_relevance = await matcher.sort_works_by_relevance(all_top_works, abstract13)
    top_referees = await matcher.get_top_referees(sorted_works_by_relevance, from_year=2016, min_citations=10, max_citations=50)
    
    top_referee_works = matcher.extract_batched_works(top_referees)
    referees_rej_works = await matcher.reject_irrelevant_works_from_referees(top_referee_works, abstract=abstract13)

    # # Save to a file
    with open("referees_rej_works.json", "w", encoding="utf-8") as f:
        json.dump(referees_rej_works, f, ensure_ascii=False, indent=2)

# Run main
asyncio.run(main())