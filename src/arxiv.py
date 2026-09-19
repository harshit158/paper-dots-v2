"""Small arXiv client used by the paper onboarding flow."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from foundry.observability import get_logger, get_tracer

from errors import ValidationError

logger = get_logger(__name__)
tracer = get_tracer(__name__)

ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org"}
ARXIV_ID_PATTERN = re.compile(r"^(?P<id>[A-Za-z0-9.-]+?)(?P<version>v\d+)?$")
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class ArxivPaper:
    title: str
    authors: str
    abstract: str


class ArxivClient:
    """Fetch metadata and PDFs from arXiv using the standard library."""

    def normalize_url(self, url: str) -> tuple[str, str]:
        parsed = urlparse(url.strip())
        if parsed.scheme != "https" or parsed.hostname not in ARXIV_HOSTS:
            raise ValidationError(
                "Enter an HTTPS arXiv URL, such as https://arxiv.org/abs/1706.03762."
            )

        match = re.fullmatch(r"/(?:abs|pdf)/([^/?#]+?)(?:\.pdf)?", parsed.path)
        if match is None:
            raise ValidationError("Use an arXiv abstract or PDF URL, such as /abs/1706.03762.")

        identifier = match.group(1)
        if ARXIV_ID_PATTERN.fullmatch(identifier) is None:
            raise ValidationError("That does not look like a valid arXiv identifier.")
        return f"https://arxiv.org/abs/{identifier}", identifier

    def fetch_metadata(self, arxiv_id: str) -> ArxivPaper:
        logger.info("fetching arXiv metadata", extra={"arxiv_id": arxiv_id})
        request = Request(
            f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
            headers={"User-Agent": "PaperDots/0.1"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                root = ET.fromstring(response.read())
        except (HTTPError, URLError, TimeoutError, ET.ParseError) as error:
            raise ValidationError(
                "Could not fetch paper details from arXiv. Please try again."
            ) from error

        entry = root.find(f"{ATOM}entry")
        if entry is None:
            raise ValidationError("arXiv could not find that paper.")

        def text(tag: str) -> str:
            value = entry.findtext(f"{ATOM}{tag}")
            return " ".join((value or "").split())

        title = text("title")
        authors = ", ".join(
            " ".join((author.findtext(f"{ATOM}name") or "").split())
            for author in entry.findall(f"{ATOM}author")
        )
        abstract = text("summary")
        if not title:
            raise ValidationError("arXiv returned incomplete paper details.")
        return ArxivPaper(title=title, authors=authors, abstract=abstract)

    def download_pdf(self, arxiv_id: str, destination: Path) -> None:
        logger.info("downloading arXiv PDF", extra={"arxiv_id": arxiv_id})
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(
            f"https://arxiv.org/pdf/{arxiv_id}",
            headers={"User-Agent": "PaperDots/0.1"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            raise ValidationError("Could not download the paper PDF from arXiv.") from error
        if not content.startswith(b"%PDF"):
            raise ValidationError("arXiv returned an invalid PDF.")
        destination.write_bytes(content)
        logger.info("arXiv PDF downloaded", extra={"arxiv_id": arxiv_id})
