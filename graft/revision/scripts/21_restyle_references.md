# revision/scripts/21_restyle_references.py

- norm_title · function · L35-L38 — def norm_title(s: str) -> str
- crossref · function · L41-L52 — def crossref(url: str) -> dict | None
- by_doi · function · L55-L56 — def by_doi(doi: str) -> dict | None
- by_title · function · L59-L72 — def by_title(title: str, author: str) -> dict | None
- year_of · function · L75-L80 — def year_of(rec: dict) -> int | None
- format_authors · function · L83-L108 — def format_authors(rec: dict) -> str | None
- journal_abbrev · function · L111-L115 — def journal_abbrev(rec: dict) -> str
- format_entry · function · L118-L142 — def format_entry(rec: dict) -> str | None
- parse_refs · function · L148-L150 — def parse_refs(text: str) -> list[str]
- extract_doi · function · L153-L155 — def extract_doi(entry: str) -> str | None
- extract_title · function · L158-L162 — def extract_title(entry: str) -> str
- first_surname · function · L165-L166 — def first_surname(entry: str) -> str
- main · function · L169-L235 — def main() -> int
