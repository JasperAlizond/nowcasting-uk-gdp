import requests
import time
from typing import List, Dict


class ConceptExplorer:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.headers = {"User-Agent": f"mailto:{email}"}

    def search_concepts(self, search_term: str) -> List[Dict]:
        """Search for concepts using a search term"""
        params = {"search": search_term, "per-page": 10}
        response = requests.get(
            f"{self.base_url}/concepts", headers=self.headers, params=params
        )
        response.raise_for_status()
        return response.json().get("results", [])

    def verify_concept(self, concept_id: str) -> Dict:
        """Verify a concept ID exists and return its details"""
        # Remove the base URL if it's included
        concept_id = concept_id.replace("https://openalex.org/", "")
        response = requests.get(
            f"{self.base_url}/concepts/{concept_id}", headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def check_concept_usage(self, concept_id: str) -> int:
        """Check how many works are associated with this concept"""
        # Remove the base URL if it's included
        concept_id = concept_id.replace("https://openalex.org/", "")
        params = {"filter": f"concepts.id:{concept_id}", "per-page": 1}
        response = requests.get(
            f"{self.base_url}/works", headers=self.headers, params=params
        )
        response.raise_for_status()
        return response.json().get("meta", {}).get("count", 0)


def main():
    # Initialize explorer with your email
    explorer = ConceptExplorer("your.email@example.com")

    # 1. First, let's search for relevant economic concepts
    search_terms = [
        "economic conditions",
        "macroeconomics",
        "business cycle",
        "economic growth",
        "economic indicator",
    ]

    print("Searching for relevant concepts:")
    print("-" * 50)
    for term in search_terms:
        print(f"\nResults for '{term}':")
        concepts = explorer.search_concepts(term)
        for concept in concepts:
            print(f"\nID: {concept['id']}")
            print(f"Name: {concept['display_name']}")
            print(f"Description: {concept.get('description', 'No description')}")
            try:
                work_count = explorer.check_concept_usage(concept["id"])
                print(f"Number of works: {work_count:,}")
            except Exception as e:
                print(f"Error checking work count: {str(e)}")
        time.sleep(1)  # Be nice to the API

    # 2. Then verify the specific IDs we were using
    print("\n\nVerifying previously used concept IDs:")
    print("-" * 50)
    concept_ids = [
        "C41008148",  # "Economic Conditions"
        "C138885662",  # "Business cycle"
        "C144133560",  # "Macroeconomics"
        "C86803240",  # "Economic growth"
        "C127313418",  # "Economic indicator"
        "C199360897",  # "Economic analysis"
    ]

    for concept_id in concept_ids:
        try:
            concept = explorer.verify_concept(concept_id)
            print(f"\nID: {concept['id']}")
            print(f"Name: {concept['display_name']}")
            print(f"Description: {concept.get('description', 'No description')}")
            work_count = explorer.check_concept_usage(concept["id"])
            print(f"Number of works: {work_count:,}")
        except requests.exceptions.RequestException as e:
            print(f"Error verifying concept {concept_id}: {str(e)}")
        time.sleep(1)  # Be nice to the API


if __name__ == "__main__":
    main()
