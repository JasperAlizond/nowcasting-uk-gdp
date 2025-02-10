import requests
import time
import json
import os
from typing import List, Dict, Optional


class OpenAlexClient:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.headers = {"User-Agent": f"mailto:{email}"}
        # Rate limits as per documentation
        self.requests_per_second = 10
        self.last_request_time = 0

    def _handle_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 1.0 / self.requests_per_second:
            time.sleep((1.0 / self.requests_per_second) - time_since_last_request)
        self.last_request_time = time.time()

    def get_concept_id(self, concept_name: str) -> Optional[str]:
        params = {"filter": f"display_name.search:{concept_name}", "per-page": 1}
        try:
            self._handle_rate_limit()
            response = requests.get(
                f"{self.base_url}/concepts", headers=self.headers, params=params
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0].get("id")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting concept ID: {str(e)}")
            return None

    def _get_page_with_cursor(self, base_url: str, params: Dict) -> tuple:
        """Helper method to handle cursor-based pagination"""
        self._handle_rate_limit()
        response = requests.get(base_url, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", []), data.get("meta", {}).get("next_cursor")

    def get_economics_abstracts_by_date_range(
        self, start_year: int, end_year: int, total_target_size: int = 100000
    ) -> List[Dict]:
        """
        Fetch economics papers across multiple concepts, using proportional sampling.

        Args:
            start_year: Starting year for data collection
            end_year: Ending year for data collection
            total_target_size: Target total number of papers to collect (default: 100,000)
        """
        # Define concepts with their IDs and approximate total paper counts
        concepts = [
            {
                "id": "https://openalex.org/C139719470",
                "name": "Macroeconomics",
                "count": 2557910,
            },
            {
                "id": "https://openalex.org/C50522688",
                "name": "Economic growth",
                "count": 3574627,
            },
            {
                "id": "https://openalex.org/C83873408",
                "name": "Business cycle",
                "count": 40636,
            },
            {
                "id": "https://openalex.org/C195742910",
                "name": "Recession",
                "count": 73381,
            },
            {
                "id": "https://openalex.org/C202353208",
                "name": "Economic indicator",
                "count": 7978,
            },
            {
                "id": "https://openalex.org/C55986821",
                "name": "Macroeconomic model",
                "count": 2854,
            },
            {
                "id": "https://openalex.org/C181683161",
                "name": "Real GDP",
                "count": 16087,
            },
        ]

        total_papers = sum(c["count"] for c in concepts)
        all_papers = []

        for concept in concepts:
            # Calculate proportional sample size for this concept
            concept_proportion = concept["count"] / total_papers
            concept_sample_size = min(
                int(total_target_size * concept_proportion), 10000
            )  # OpenAlex sample limit

            print(
                f"\nProcessing {concept['name']} (targeting {concept_sample_size:,} papers)..."
            )

            try:
                # Base parameters for this concept
                base_params = {
                    "filter": f"concepts.id:{concept['id']},from_publication_date:{start_year}-01-01,to_publication_date:{end_year}-12-31",
                    "select": "id,title,publication_date,doi,abstract_inverted_index",
                    "per-page": 200,  # Maximum allowed per page
                    "sample": concept_sample_size,
                    "seed": 42,  # Fixed seed for reproducibility
                }

                # Get sampled papers for this concept
                results = []
                page = 1
                while True:
                    self._handle_rate_limit()
                    response = requests.get(
                        f"{self.base_url}/works",
                        headers=self.headers,
                        params={**base_params, "page": page},
                    )
                    response.raise_for_status()
                    data = response.json()
                    batch = data.get("results", [])
                    if not batch:
                        break

                    # Process papers and reconstruct abstracts
                    for paper in batch:
                        abstract_index = paper.get("abstract_inverted_index")
                        if abstract_index:
                            words = []
                            for word, positions in abstract_index.items():
                                for pos in positions:
                                    while len(words) <= pos:
                                        words.append("")
                                    words[pos] = word
                            abstract = " ".join(words).strip()

                            all_papers.append(
                                {
                                    "title": paper.get("title"),
                                    "abstract": abstract,
                                    "publication_date": paper.get("publication_date"),
                                    "doi": paper.get("doi"),
                                    "id": paper.get("id"),
                                    "concept": concept[
                                        "name"
                                    ],  # Add concept for reference
                                }
                            )

                    print(
                        f"Retrieved {len(batch)} papers (total for {concept['name']}: {len(results)})"
                    )
                    results.extend(batch)

                    # Check if we've hit the sample size
                    if len(results) >= concept_sample_size:
                        break

                    page += 1

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for concept {concept['name']}: {str(e)}")
                continue

            print(f"Completed {concept['name']}: retrieved {len(results):,} papers")

        print(f"\nTotal papers collected across all concepts: {len(all_papers):,}")
        return all_papers


def save_abstracts_to_file(
    papers: List[Dict], filename: str = "data/economics_abstracts_historical.json"
):
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)


# Usage example
if __name__ == "__main__":
    client = OpenAlexClient(email="your.email@example.com")  # Replace with your email
    papers = client.get_economics_abstracts_by_date_range(
        start_year=1998, end_year=2024
    )

    print("\nSummary:")
    print(f"Found total of {len(papers)} papers with abstracts")

    if papers:
        print("\nExample of first paper:")
        print(f"Title: {papers[0]['title']}")
        print(f"Abstract preview: {papers[0]['abstract'][:200]}...")

    save_abstracts_to_file(papers)
    print("Saved results to economics_abstracts_historical.json")
