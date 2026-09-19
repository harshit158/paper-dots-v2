from pathlib import Path

import pytest

from arxiv import ArxivClient, ArxivPaper
from errors import ValidationError


def test_normalize_abs_url():
    assert ArxivClient().normalize_url(" https://arxiv.org/abs/1706.03762 ") == (
        "https://arxiv.org/abs/1706.03762",
        "1706.03762",
    )


def test_normalize_pdf_url():
    assert ArxivClient().normalize_url("https://arxiv.org/pdf/1706.03762.pdf")[1] == "1706.03762"


@pytest.mark.parametrize(
    "url", ["", "https://example.com/abs/1706.03762", "https://arxiv.org/foo/1706.03762"]
)
def test_rejects_non_arxiv_urls(url):
    with pytest.raises(ValidationError):
        ArxivClient().normalize_url(url)


def test_import_fetches_metadata_and_pdf(papers, tmp_path):
    class FakeArxiv:
        def normalize_url(self, url):
            return "https://arxiv.org/abs/1706.03762", "1706.03762"

        def fetch_metadata(self, arxiv_id):
            return ArxivPaper("Attention Is All You Need", "Ashish Vaswani", "Abstract")

        def download_pdf(self, arxiv_id, destination: Path):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"%PDF-fake")

    from services import PaperService

    paper, already_existed = PaperService(papers._repository, FakeArxiv()).import_arxiv_paper(
        "https://arxiv.org/abs/1706.03762", tmp_path
    )

    assert not already_existed
    assert paper.title == "Attention Is All You Need"
    assert paper.authors == "Ashish Vaswani"
    assert Path(paper.pdf_path).read_bytes() == b"%PDF-fake"
