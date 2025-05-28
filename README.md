backend/
│
├── .venv/              ← your virtual environment (do not put code here)
├── app/                ← your backend application files
│   ├── main.py         ← FastAPI entry point
│   └── routes/         ← optional subfolder for route files
│
├── requirements.txt    ← optional: save dependencies here
├── README.md

Algorithm for editor finder

As far as I know, open alex provides up to 3 topics, and 5 keywords for each work. Each topic and keyword has a score. Additionally it lets you have the primary topic in a separate field. I have written an efficient algorithm to find best matched authors:

DONE 1. Find 3 ranked topics (abstract_topics) and 5 ranked keywords (abstract_kws) from the abstraction - both must be sorted by score
DONE 2. Filter works having topics found in step (1) as their primary topic
DONE 3.1 Extract all distinct primary keywords, those with highest score, from works - use set() to make sure keywords are distinct; it can be named works_primary_kws
DONE 3.2 For each keyword in step (1) do the following:
3.2.1 Find the primary match using sentence transformer
3.2.2 Store the first match in a first_primary_kw_list
3.2.3 Store the second match in a second_primary_kw_list
3.2.4 Store the third match in a third_primary_kw_list
3.2.5 Store the fourth match in a fourth_primary_kw_list
3.2.6 Store the fifth match in a fifth_primary_kw_list
DONE 4. Embed abstract_kws onto works_primary_kws. Find 3 matches for the first keyword, 2 matches for the second keyword, and one match for each remaining keyword. 
5. Use works found in step (2) and keywords found in step (4) to find authors of the works. You know several works may have the same keyword.
6. We should only keep the first two primary authors of each work, as they are strongly knowledgeable than the other co-authors if exist.
7. If the number of authors is less than a minimum, try to use the second_primary_kw_list to find new works, and then new authors. You need to repeat steps (5) and (6) until you get enough authors to display.
8. Now we have found quite a lot of promising authors.
9. Rank authors found in step (8) by
9.1 primary keywords density throughout their works
9.2 Number of matching papers
9.3 Recency of publications
9.4 Whether the keyword appears in multiple of his papers (a measure of his research continuity)
10. Filter ranked authors down to those not having conflicts with the user. You know some of the authors may have previously worked with him, or studied in the same institution. Those must be filtered out.
11. At this point we have got a list of authors who are what we call "best editor matches"!

deepseek api key
sk-066c15c1bb7f4539bb20b4a9952b25a9