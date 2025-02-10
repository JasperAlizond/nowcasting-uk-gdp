import json
from collections import Counter
import re
import statistics


def analyze_papers(filename="economics_abstracts_historical.json"):
    """Analyze the economics papers dataset and print summary statistics."""

    # Load the JSON file
    with open(filename, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Basic statistics
    total_papers = len(papers)

    # Analyze publication dates
    dates = [paper["publication_date"] for paper in papers if paper["publication_date"]]
    months = Counter([date[:7] for date in dates])  # Get YYYY-MM

    # Analyze abstract lengths
    abstract_lengths = [
        len(paper["abstract"].split()) for paper in papers if paper["abstract"]
    ]

    # Analyze common words in titles (excluding common stop words)
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
    }
    title_words = []
    for paper in papers:
        if paper["title"]:
            words = re.findall(r"\w+", paper["title"].lower())
            title_words.extend([w for w in words if w not in stop_words and len(w) > 2])

    print("\n=== Economics Papers Dataset Analysis ===")
    print(f"\nTotal number of papers: {total_papers}")

    print("\nAbstract Length Statistics:")
    print(f"Average words per abstract: {statistics.mean(abstract_lengths):.1f}")
    print(f"Median words per abstract: {statistics.median(abstract_lengths):.1f}")
    print(f"Shortest abstract: {min(abstract_lengths)} words")
    print(f"Longest abstract: {max(abstract_lengths)} words")

    print("\nTop 10 Most Common Words in Titles:")
    for word, count in Counter(title_words).most_common(10):
        print(f"{word}: {count} occurrences")

    print("\nPublication Date Distribution:")
    print("Month          Papers")
    print("-" * 25)
    for month, count in sorted(months.items(), reverse=True)[:6]:  # Show last 6 months
        print(f"{month}:    {count}")

    # Calculate papers with DOIs
    papers_with_doi = sum(1 for paper in papers if paper.get("doi"))
    doi_percentage = (papers_with_doi / total_papers) * 100
    print(f"\nPapers with DOIs: {papers_with_doi} ({doi_percentage:.1f}%)")

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get the first entry as an example
    example_entry = data[0] if data else None

    print("\n=== JSON Structure Analysis ===")
    print("\nOverall Structure:")
    print("Root level: List of paper objects")
    print(f"Number of papers: {len(data)}")

    if example_entry:
        print("\nFields in each paper object:")
        for key, value in example_entry.items():
            field_type = type(value).__name__
            example_val = str(value)
            if len(example_val) > 50:
                example_val = example_val[:50] + "..."
            print(f"- {key} ({field_type}):")
            print(f"  Example: {example_val}")

    print("\nComplete example entry (first paper):")
    if example_entry:
        print(json.dumps(example_entry, indent=2))


if __name__ == "__main__":
    analyze_papers()
