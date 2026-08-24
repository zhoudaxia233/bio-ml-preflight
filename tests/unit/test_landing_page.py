from hashlib import sha256
from pathlib import Path


def test_landing_stylesheet_url_matches_content_hash() -> None:
    root = Path(__file__).parents[2]
    css = (root / "docs" / "landing.css").read_bytes()
    html = (root / "docs" / "index.html").read_text()

    assert f'href="./landing.css?v={sha256(css).hexdigest()[:12]}"' in html
