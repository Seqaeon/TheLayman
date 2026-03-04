import json
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from math import log1p

from the_layman.backend.schemas import DailyFeedItem, PaperScore, UserPreferences
from the_layman.database.store import Store
from the_layman.pipeline.buzz import fetch_buzz_scores
from the_layman.pipeline.ingestion import _http_get, _normalize_text, _parse_arxiv_abs_html
from the_layman.pipeline.llm_client import generate_json


def fetch_recent_arxiv_papers(target_fields: list[str], max_results: int = 500) -> list[dict]:
    """Fetch recent papers from arXiv for the given fields."""
    if not target_fields:
        return []

    # Build a query: e.g. (cat:cs.AI OR cat:cs.LG)
    query_parts = [f"cat:{field.strip()}" for field in target_fields if field.strip()]
    search_query = f"({' OR '.join(query_parts)})"
    
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results
    }
    url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    print(f"Fetching from arXiv URL: {url}")
    
    papers = []
    try:
        xml_data = _http_get(url).decode("utf-8", errors="ignore")
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_data)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        
        for entry in root.findall("a:entry", ns):
            id_url = entry.findtext("a:id", default="", namespaces=ns)
            title = _normalize_text(entry.findtext("a:title", default="", namespaces=ns))
            summary = _normalize_text(entry.findtext("a:summary", default="", namespaces=ns))
            
            # extract field category
            primary_cat = entry.find("a:category", ns)
            field = primary_cat.attrib.get("term", "unknown") if primary_cat is not None else "unknown"

            if id_url and title:
                # get plain ID
                arxiv_id = id_url.split('/abs/')[-1]
                papers.append({
                    "id": f"arxiv:{arxiv_id}",
                    "title": title,
                    "abstract": summary,
                    "field": field
                })
    except Exception as e:
        print(f"Error fetching from arXiv: {e}")
        
    return papers


def score_keywords(papers: list[dict], priority_keywords: list[str]) -> None:
    """Adds a 'keyword_score' to each paper dict based on presence of keywords (case-insensitive)."""
    if not priority_keywords:
        for p in papers:
            p["keyword_score"] = 0
        return
        
    keywords_lower = [k.lower() for k in priority_keywords if k.strip()]
    
    for paper in papers:
        score = 0
        text_to_search = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        for kw in keywords_lower:
            if kw in text_to_search:
                # +10 points per matched keyword
                score += 10
                
        paper["keyword_score"] = score


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def score_llm_relevance(papers: list[dict], instruction: str) -> None:
    """Uses the LLM to assign an impact score (1-10) to each paper in batches."""
    if not papers:
        return
        
    # We batch to avoid breaking context limits and recency bias. 50 abstracts is a safe size.
    batches = list(_batch(papers, 50))
    print(f"Scoring {len(papers)} papers in {len(batches)} batches using LLM...")
    
    for batch_ndx, batch in enumerate(batches):
        print(f"Processing batch {batch_ndx + 1}/{len(batches)}...")
        input_data = [{"id": p["id"], "title": p["title"], "abstract": p.get("abstract", "")[:800]} for p in batch]
        
        prompt = f"""You are an expert scientific evaluator. 
I am going to provide you with a list of academic papers (ID, Title, Abstract) enclosed in <papers> tags.

Your task is to assign a Real-World Impact Score from 1-10 to each paper based strictly on the following Relevance Instruction:
"{instruction}"

<papers>
{json.dumps(input_data, indent=2)}
</papers>

You must return strictly valid JSON. 
To ensure reasoning, the first key in your JSON MUST be "reasoning_scratchpad", containing a brief summary of how you applied the instruction to this batch.
The second key MUST be "scores", which is an array of objects containing the "id" and the integer "impact_score".

Example Output Format:
{{
  "reasoning_scratchpad": "Applied the instruction by highly scoring papers that directly affect medicine, while down-ranking theoretical math papers.",
  "scores": [
    {{"id": "arxiv:1234.5678", "impact_score": 8}},
    {{"id": "arxiv:9876.5432", "impact_score": 3}}
  ]
}}
"""
        result = generate_json(prompt)
        
        # map scores back
        score_map = {}
        if result and "scores" in result and isinstance(result["scores"], list):
            for s in result["scores"]:
                if isinstance(s, dict) and "id" in s and "impact_score" in s:
                    try:
                        score_map[s["id"]] = int(s["impact_score"])
                    except (ValueError, TypeError):
                        pass

        for paper in batch:
            paper["llm_impact_score"] = score_map.get(paper["id"], 5)  # Default 5 if model skips it


def generate_daily_feed(store: Store) -> None:
    print("Starting daily feed generation...")
    prefs = store.get_user_preferences()
    print(f"User Preferences: fields={prefs.target_fields}, keywords={prefs.priority_keywords}")
    target_fields = prefs.target_fields or ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
    
    # 1. Fetch papers from arXiv
    print(f"Fetching recent papers for fields: {target_fields}")
    papers = fetch_recent_arxiv_papers(target_fields, max_results=100)
    print(f"Fetched {len(papers)} papers.")
    
    if not papers:
        print("No papers found. Aborting.")
        return

    # 2. Fast pass — keyword scoring
    score_keywords(papers, prefs.priority_keywords)
    
    # Pre-sort and trim to top 150 to save LLM tokens
    papers.sort(key=lambda x: x["keyword_score"], reverse=True)
    papers_to_score = papers[:150]

    # 3. Community buzz scoring (HN + Semantic Scholar)
    print(f"Fetching community buzz for {len(papers_to_score)} papers...")
    paper_ids = [p["id"] for p in papers_to_score]
    buzz_scores = fetch_buzz_scores(paper_ids)
    for p in papers_to_score:
        p["buzz_score"] = buzz_scores.get(p["id"], 0.0)

    # 4. Smart pass — LLM impact scoring
    score_llm_relevance(papers_to_score, prefs.relevance_instruction)
    
    # 5. Final ranking
    # Weights: base 25% | keyword 20% | LLM 35% | buzz 20%
    # (previously base 30% | keyword 30% | LLM 40%)
    paper_scores_db = []
    daily_feed_items = []
    
    for p in papers_to_score:
        k_score  = p.get("keyword_score", 0)
        l_score  = p.get("llm_impact_score", 5)
        b_score  = p.get("buzz_score", 0.0)
        
        normalized_k = min(k_score / 50.0, 1.0) * 10
        total = (0.25 * 10) + (0.20 * normalized_k) + (0.35 * l_score) + (0.20 * b_score)
        p["total_score"] = total
        
        paper_scores_db.append(PaperScore(
            paper_id=p["id"],
            keyword_score=k_score,
            llm_impact_score=l_score,
            buzz_score=b_score,
            total_score=total,
        ))

    # Sort final
    papers_to_score.sort(key=lambda x: x["total_score"], reverse=True)
    top_100 = papers_to_score[:100]
    
    for p in top_100:
        daily_feed_items.append(
            DailyFeedItem(
                paper_id=p["id"],
                title=p["title"],
                field=p["field"],
                impact_score=p["llm_impact_score"],
                buzz_score=p.get("buzz_score", 0.0),
                abstract_preview=p["abstract"][:200] + "..." if len(p.get("abstract", "")) > 200 else p.get("abstract", "")
            )
        )
        
    # 6. Save
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    store.save_paper_scores(paper_scores_db)
    store.save_daily_feed(date_str, daily_feed_items)
    
    print(f"Successfully generated and saved feed for {date_str} with {len(top_100)} items.")
