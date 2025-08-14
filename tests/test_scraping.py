import bs4
from scraping import is_valid_text, sanitize_filename, get_parent_domain, extract_links

def test_is_valid_text_rejects_garbled():
    assert not is_valid_text("��������")
    assert not is_valid_text("N/A")
    assert not is_valid_text("x")  

def test_is_valid_text_accepts_normal():
    assert is_valid_text("This is a valid line of text with ASCII.")

def test_sanitize_filename():
    assert sanitize_filename('bad:name?.txt') == 'bad_name_.txt'

def test_get_parent_domain():
    assert get_parent_domain("https://sub.uni.example.co.uk/path") == "example.co.uk"

def test_extract_links_filters_social_and_files():
    html = """
      <a href="/a.pdf">PDF</a>
      <a href="https://twitter.com/x">tw</a>
      <a href="https://site.edu/program">ok</a>
      <a href="/img.png">img</a>
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    links = extract_links(soup, "https://site.edu")
    assert links == {"https://site.edu/program"}
