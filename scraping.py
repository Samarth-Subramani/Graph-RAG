import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException
import fitz  # PyMuPDF for PDF processing
import os
import re
import csv
import json
import PyPDF2
import hashlib
import tldextract
from bs4.exceptions import ParserRejectedMarkup
import io

# Directory for saving data
base_data_dir = "final_data"
os.makedirs(base_data_dir, exist_ok=True)

# File to store visited URLs
visited_urls_log = "visited_urls.log"
skipped_urls_log = "skipped_urls.log"
prioritized_links = set()

# Load visited or skipped URLs if the file exists
if os.path.exists(visited_urls_log):
    with open(visited_urls_log, "r") as f:
        visited_urls = set(json.load(f))
else:
    visited_urls = set()

if os.path.exists(skipped_urls_log):
    with open(skipped_urls_log, "r", encoding="utf-8") as f:
        skipped_urls = {line.split(" | ")[0] for line in f}
else:
    skipped_urls = set()

def log_visited_urls():
    """Save visited URLs to a JSON file."""
    with open(visited_urls_log, "w") as f:
        json.dump(list(visited_urls), f)

def log_skipped_url(url, reason):
    with open(skipped_urls_log, "a", encoding="utf-8") as f:
        f.write(f"{url} | Reason: {reason}\n")

def get_university_dir(url):

    parent_domain = get_parent_domain(url)
    # Use the parent domain to create a directory name
    directory_name = parent_domain.split('.')[0]  # Extract the second-level domain only (e.g., "sorbonne")
    university_dir = os.path.join(base_data_dir, directory_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(university_dir, exist_ok=True)
    
    return university_dir


user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.54',
    'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.210 Mobile Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
    'Mozilla/5.0 (Linux; Android 10; SM-G960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Mobile Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 13_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 8.1.0; Nexus 6P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.80 Mobile Safari/537.36',
]

headers = {
    'User-Agent': random.choice(user_agents)
}

session = requests.Session()
session.headers.update(headers)

proxies = None

# Generate a short filename from URL hash
def hash_filename(url):
    hash_digest = hashlib.sha256(url.encode()).hexdigest()
    return f"url_{hash_digest[:16]}.txt"  # Use first 16 chars of hash

# Heuristic Validity Check 
def is_valid_text(text):
    if not text.strip():    # check empty files
        return False
    if len(text.strip()) < 10:  # check if file length less than 10
        return False
    if text.strip().lower() in {"n/a", "na", "none", "null"}:   # check if file has only the following values
        return False
    if len(set(text.strip())) == 1:
        return False
    if re.search(r'[^\x00-\x7F]{4,}', text):  # e.g., ����
        return False
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

def is_valid_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = content.strip()

            if not cleaned:
                return False
            if len(cleaned) < 10:
                return False
            if cleaned.lower() in {"n/a", "na", "none", "null"}:
                return False
            if len(set(cleaned)) == 1:
                return False
            if re.search(r'[^\x00-\x7F]{4,}', cleaned):  # 4+ consecutive non-ASCII (often garbled)
                return False
            if re.search(r'[@#%$^*+=~_\\\-]{6,}', cleaned):
                return False
            cleaned.encode('utf-8').decode('utf-8')  # Check UTF-8 decodability
            return True
    except Exception as e:
        print(f"⚠️ CSV error in {file_path}: {e}")
        return False

def save_metadata(university_dir, filename, url, content_path):
    meta_path = os.path.join(university_dir, "metadata.json")  # single metadata file for the entire university_dir

    # Load existing metadata if available
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                metadata_all = json.load(f)
                if not isinstance(metadata_all, dict):
                    print(f"Warning: {meta_path} is not a dict. Resetting metadata.")
                    metadata_all = {}
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {meta_path}. Resetting metadata.")
                metadata_all = {}
    else:
        metadata_all = {}

    # Add/update metadata for this file
    metadata_all[filename] = {
        "source_url": url,
        "content_file": content_path
    }

    # Save updated metadata back to file
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_all, f, indent=2)

    # Save the combined metadata back to file
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_all, f, indent=2)
        
def fetch_page_content(url):
    """Fetches and parses HTML content from a URL."""
    try:
        response = session.get(url, proxies=proxies)
        response.raise_for_status()  # Raises HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        if soup:
            return soup
        else:
            return BeautifulSoup(response.content, 'lxml')
        
    except (SSLError, ConnectionError, Timeout) as e:
        print(f"[SKIPPED] {url} due to connection error: {e}")
        log_skipped_url(url, f"Connection error: {e}")
    except RequestException as e:
        print(f"[SKIPPED] {url} due to request exception: {e}")
        log_skipped_url(url, f"RequestException: {e}")
    except ParserRejectedMarkup as e:
        print(f"[SKIPPED] {url} due to BeautifulSoup parsing error: {e}")
        log_skipped_url(url, f"ParserRejectedMarkup: {e}")
    except Exception as e:
        print(f"[SKIPPED] {url} due to unknown error: {e}")
        log_skipped_url(url, f"Unknown error: {e}")
        return None

def sanitize_filename(filename):
    """Removes or replaces invalid characters in a filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def extract_text(soup, filename, university_dir, url=None):
    """Extracts text and saves it only if valid. Includes source URL."""
    print("                     ....Extracting text")
    texts = soup.get_text(separator='\n', strip=True)

    if not is_valid_text(texts):
        print("                     ❌ Skipped invalid text content.")
        log_skipped_url(url, "Invalid content based on heuristic check")
        return None  # Do not save invalid content

    sanitized_filename = sanitize_filename(filename)
    text_filename = os.path.join(university_dir, f"{sanitized_filename}_content.txt")

    with open(text_filename, 'w', encoding='utf-8') as f:
        if url:
            f.write(f"Source URL: {url}\n\n")
        f.write(texts)
    
    return text_filename

def extract_table_data(soup, filename, university_dir):
    """Extracts table data from the BeautifulSoup object and saves to CSV."""
    print("                     ....Extracting table")
    tables = soup.find_all('table')  # Find all tables
    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        table_data = []
        for row in rows:
            cols = row.find_all(['td', 'th'])
            cols = [col.get_text(strip=True) for col in cols]  # Clean up text
            table_data.append(cols)
        
        table_filename = os.path.join(university_dir, f"{filename}_table_{i}.csv")

        # Save table data to CSV
        with open(table_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_data)

        # Remove file if it's invalid
        if not is_valid_csv(table_filename):
            os.remove(table_filename)
            print(f"                     ❌ Removed invalid CSV: {table_filename}")
        else:
            print(f"                     ✅ Saved valid table: {table_filename}")

def extract_pdf_links(soup, base_url, filename, university_dir):
    """Extracts PDF links from the BeautifulSoup object and extracts their text content without saving the PDF."""
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.pdf'):
            pdf_url = urljoin(base_url, href)
            pdf_links.append(pdf_url)
    
    for i, pdf_url in enumerate(pdf_links):
        try:
            headers = {
                'User-Agent': random.choice(user_agents)
            }
            try:
                response = requests.get(pdf_url, headers=headers)
                response.raise_for_status()
            except UnicodeDecodeError as e:
                print(f"Skipping link due to UnicodeDecodeError: {pdf_url}. Error: {e}")
                continue

            if 'application/pdf' not in response.headers.get('Content-Type', '') or len(response.content) < 1024:
                continue

            # In-memory processing of the PDF using fitz
            full_text = ""
            try:
                pdf_stream = io.BytesIO(response.content)
                doc = fitz.open(stream=pdf_stream, filetype='pdf')
                for page in doc:
                    full_text += page.get_text()

                # Validate the extracted text
                if not is_valid_text(full_text):
                    print(f"                     ❌ Skipped invalid PDF content: {pdf_url}")
                    log_skipped_url(pdf_url, "Invalid PDF text content")
                    continue

                print(f"Extracted in-memory PDF content from {pdf_url}")

                # Save text content to disk
                pdf_name = f"{filename}_{i}.txt"
                pdf_name = sanitize_filename(pdf_name)
                text_path = os.path.join(university_dir, pdf_name)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(f"Source URL: {pdf_url}\n\n")
                    f.write(full_text)
                
            except fitz.FileDataError as mupdf_error:
                print(f"MuPDF error for in-memory PDF {pdf_url}: {mupdf_error}")
                print(f"Trying PyPDF2 in-memory for: {pdf_url}")
                text = read_pdf_with_pypdf2_bytes(response.content)
                # Validate the fallback text
                if text and is_valid_text(text):
                    pdf_name = f"{filename}_{i}_pypdf2.txt"
                    text_path = os.path.join(university_dir, sanitize_filename(pdf_name))
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(f"Source URL: {pdf_url}\n\n")
                        f.write(text)
                else:
                    print(f"                     ❌ Skipped invalid fallback PDF content: {pdf_url}")
                    log_skipped_url(pdf_url, "Invalid fallback PDF content or empty")
                    with open(os.path.join(university_dir, "failed_pdfs.log"), 'a', encoding='utf-8') as log_file:
                        log_file.write(f"Failed to process PDF: {pdf_url}\n")
        except (requests.RequestException, fitz.FileDataError, OSError) as e:
            print(f"Failed to download or process PDF: {pdf_url}. Error: {e}")
            with open(os.path.join(university_dir, "failed_pdfs.log"), 'a', encoding='utf-8') as log_file:
                log_file.write(f"Failed to download or process PDF: {pdf_url}\nError: {e}\n")

def read_pdf_with_pypdf2_bytes(pdf_bytes):
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        print(f"Error processing in-memory PDF. Error: {e}")
        return None
    

def extract_links(soup, base_url):
    """Extracts all links from the page to follow and scrape further, excluding social media profiles."""
    
    # List of social media URLs to exclude
    social_media_urls = [
        'facebook.com', 'linkedin.com', 'twitter.com', 'instagram.com', 'youtube.com', 'zoom.com', 'eurohpc-ju.europa.eu',
        'pinterest.com', 'tiktok.com', 'whatsapp.com', 'wechat.com', 'snapchat.com', 'outlook.com', 'europa.eu', 'feeds', 'news', 'google','x', 'zoom', 'bsky', 'reddit.com', 'tumblr.com', 'vk.com', 'wordpress.com', 'ec.europa.eu', 'european-union.europa.eu','orbilu.uni.lu', 'wikipedia.org',
    ]

    links = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        
        # Exclude PDF, .jpg, .jpeg , .png and .webp links and social media references 
        if not href.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.webp')) and is_valid_url(full_url):
            if not any(social_media_url in full_url for social_media_url in social_media_urls) and len(full_url) < 250:
                links.add(full_url)
    
    return links


def is_valid_url(url):
    """Checks if a URL is valid (allows different domains)."""
    parsed_url = urlparse(url)
    return parsed_url.scheme in ('https')

def get_parent_domain(url):
    """Extract and return the parent domain or relevant identifier for a given URL."""
    # Parse the URL
    extracted = tldextract.extract(url)
    
    # Combine the domain and suffix (e.g., 'example.com', 'co.uk')
    parent_domain = f"{extracted.domain}.{extracted.suffix}"
    
    return parent_domain

def scrape_urls(start_urls, max_depth):
    global visited_urls
    global skipped_urls

    """Iteratively scrapes URLs from a list of starting URLs, limiting depth."""
    to_visit = [(url, 0) for url in [start_urls]]   # Stack of URLs with their depth

    while to_visit:
        url, depth = to_visit.pop()  # Get the next URL and its depth
        if url in visited_urls or url in skipped_urls:
            print(f"Already visited: {url}")
            continue

        print(f"Scraping URL: {url} at depth {depth}")
        visited_urls.add(url)
        log_visited_urls()

        soup = fetch_page_content(url)


        if soup:
            # index_prefix = urlparse(url).path.replace('/', '_').strip('_')
            filename = hash_filename(url)
            university_dir = get_university_dir(url)

            # Extract and save all textual content
            content_path = extract_text(soup, filename, university_dir,url)
            if content_path:
                save_metadata(university_dir, filename, url, content_path)
            else:
                continue
            extract_table_data(soup, filename, university_dir)
            extract_pdf_links(soup, url, filename, university_dir)

            # Only add new links if we're below the max depth
            if depth < max_depth:
                new_links = extract_links(soup, url)

                to_visit.extend((new_link, depth + 1) for new_link in new_links if new_link not in visited_urls and get_parent_domain(new_link) in parent_domains)
        else:
            print(f"Failed to fetch or parse URL: {url}")
            continue  # Skip to the next URL in the stack

    log_visited_urls()

# URLs to start scraping
first_url = "https://eumaster4hpc.eu/"

parent_domains = set()
print("Extracting parent domains for initial set of links...............")
soup = fetch_page_content(first_url)
if soup:
    initial_links = extract_links(soup, first_url)

for url in initial_links:
    parent_domains.add(get_parent_domain(url))
    print(parent_domains)
    scrape_urls(url, max_depth=10)  # Set depth limit 

print("Website scraped successfully.....")
