import requests, httpx
import asyncio
import google.generativeai as genai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
import re
from typing import List, Dict, Optional
from pathlib import Path

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
  
    def extract_json_array(self, text: str) -> str:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text  # fallback to original if no brackets found

    async def get_best_matching_fields(self, abstract: str) -> Dict:
        """
        Use LLM to find the most relevant field from OpenAlex metadata based on the abstract.
        """
        with open(self.subfields, "r", encoding="utf-8") as f:      
            subfields = json.load(f)

        prompt = f"""
            You are a research field classification expert. Your task is to analyze scientific abstracts and determine the **most appropriate academic subfield** for each one. You will be given:
            - A list of candidate subfields (with their names and OpenAlex IDs),
            - A research abstract.

            Your goal is to select the **three most relevant subfields** based on the abstract’s core contribution, scientific context, and application area. Avoid choosing subfields just because of repeated keywords. Instead, focus on the **purpose of the research**, the **domain it contributes to**, and **who would most likely read or cite the paper**.

            Respond only with a JSON array of three objects, each in this format:
            {{
                "selected_subfield_name": "Subfield Name",
                "selected_subfield_id": "https://openalex.org/subfields/xxxx",
                "reason": "A short explanation for why this subfield is the best match."
            }}

            Abstract:
            \"\"\"
            {abstract}
            \"\"\"

            Candidate subfields:
            {subfields}
            """

        response = self.llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content.strip()
        clean_answer = self.extract_json_array(answer)

        try:
            selected_subfields = json.loads(clean_answer)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("Cleaned text was:", repr(clean_answer))
            selected_subfields = []

        return selected_subfields
    
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

        response = self.llm_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_output = response.choices[0].message.content.strip()

        # Try to extract the JSON content from inside code block if present
        json_data = self.extract_json_array(raw_output)

        try:
            filtered_topics = json.loads(json_data)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("Raw LLM output:\n", raw_output)
            return []

        return filtered_topics

    def extract_topic_ids(self, subfield_topic_data: List[Dict]) -> List[str]:
        """
        Extract unique topic IDs from the AI-ranked topics under each subfield.
        This works with the structure returned by `get_topics_for_selected_subfields`.
        """
        topic_ids = set()
        for subfield in subfield_topic_data:
            selected_topics = subfield.get("selected_topics", [])
            for topic in selected_topics:
                topic_id = topic.get("topic_id")
                if topic_id:
                    topic_ids.add(topic_id)
        return list(topic_ids)

    async def fetch_top_works_for_topics(
        self,
        topic_ids: List[str],
        abstract: str,
        max_works_per_topic: int = 50,
        min_publication_year: int = 2017,
        min_citations: int = 15,
        max_total_works: int = 150
    ) -> List[Dict]:
        """
        Fetch top works from OpenAlex for each topic and re-rank them using semantic similarity to the abstract.
        Applies filters: publication date, citation count, and work type.
        """
        import datetime
        import httpx

        all_works = []
        from_date = f"{min_publication_year}-01-01"
        
        async with httpx.AsyncClient() as client:
            for topic_id in topic_ids:
                url = f"{self.base_url}/works"
                params = {
                    "filter": f"topics.id:{topic_id},type:journal-article,from_publication_date:{from_date}",
                    "sort": "cited_by_count:desc",
                    "per-page": max_works_per_topic
                }

                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    works = response.json().get("results", [])
                    
                    # Additional citation filtering
                    filtered = [
                        w for w in works
                        if w.get("cited_by_count", 0) >= min_citations
                    ]
                    all_works.extend(filtered)

                except httpx.HTTPError as e:
                    print(f"Error fetching works for topic {topic_id}: {e}")

        # Limit total works before sending to AI
        all_works = all_works[:max_total_works]

        # Re-rank using LLM
        return await self.re_rank_works_by_relevance(all_works, abstract)

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
    
    abstract = 'Injection locking characteristics of oscillators are derived and a graphical analysis is presented that describes injection pulling in time and frequency domains. An identity obtained from phase and envelope equations is used to express the requisite oscillator nonlinearity and interpret phase noise reduction. The behavior of phase-locked oscillators under injection pulling is also formulated.'
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
    
    field = await matcher.get_best_matching_fields(abstract6)
    subfields_topics = await matcher.get_topics_for_selected_subfields(field)
    filters = await matcher.filter_relevant_topics_for_subfields(abstract6, subfields_topics)
    topic_ids = matcher.extract_topic_ids(filters)
    print(json.dumps(topic_ids, indent=4, ensure_ascii=False))

# Run main
asyncio.run(main())