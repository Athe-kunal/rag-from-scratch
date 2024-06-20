from pymed import PubMed
from dotenv import load_dotenv, find_dotenv
import os
from dataclasses import dataclass
from typing import Tuple

load_dotenv(find_dotenv(), override=True)


@dataclass
class PubMedMetadata:
    title: str
    authors: str
    pubmed_id: str
    keywords: str
    doi: str
    journals: str


def query_pubmed(query: str, MAX_RESULTS: int = 10) -> Tuple[str, str]:
    pubmed = PubMed(tool=os.environ["PUBMED_TOOL"], email=os.environ["PUBMED_EMAIL"])
    results = pubmed.query(query, max_results=MAX_RESULTS)
    article_abstracts = []
    article_metadata = []
    for article in results:
        article_json = article.toDict()
        article_abstracts.append(article_json["abstract"])
        curr_article_metadata = PubMedMetadata(
            title=str(article_json["title"]),
            authors=str(article_json["authors"]),
            pubmed_id=article_json["pubmed_id"],
            keywords=str(article_json["keywords"]),
            doi=str(article_json["doi"]),
            journals=str(article_json["journal"]),
        )
        article_metadata.append(curr_article_metadata.__dict__)
    return article_abstracts, article_metadata


if __name__ == "__main__":
    print(query_pubmed(query="covid-19"))
