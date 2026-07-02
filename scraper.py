import requests
from bs4 import BeautifulSoup
import json
import time
import re

BASE = "https://www.shl.com/solutions/products/product-catalog/"

def extract_test_type(card):
    # This is a placeholder since the actual structure may vary
    # E.g. test type could be in a specific span or data attribute
    type_elem = card.select_one(".test-type-badge")
    if type_elem:
        return type_elem.get_text(strip=True)
    return "Unknown"

def extract_job_levels(card):
    # Placeholder
    levels_elem = card.select_one(".job-levels")
    if levels_elem:
        text = levels_elem.get_text(strip=True)
        return [l.strip() for l in text.split(",")]
    return []

def scrape_catalog():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(BASE, headers=headers, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"Error fetching catalog: {e}")
        return

    items = []
    # Inspect the actual HTML structure first. Look for the container
    # element that holds each product card, then iterate over them.
    # The structure may be a table, a grid of <article> tags, or <li> items.
    # Adapt the selector after inspecting the page source manually.
    
    # Placeholder selector for product cards
    cards = soup.select(".product-card")
    if not cards:
        print("No product cards found with selector '.product-card'. You may need to inspect the page and update the selector.")

    for card in cards:
        name_elem = card.select_one(".product-name")
        if not name_elem: continue
        name = name_elem.get_text(strip=True)
        
        a_elem = card.select_one("a")
        url = a_elem["href"] if a_elem else ""
        if url and not url.startswith("http"):
            url = "https://www.shl.com" + url
            
        desc = card.select_one(".product-description")
        
        items.append({
            "id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
            "name": name,
            "url": url,
            "description": desc.get_text(strip=True) if desc else "",
            "test_type": extract_test_type(card),
            "job_levels": extract_job_levels(card),
            "remote_testing": bool(card.select_one(".remote-badge")),
        })
        time.sleep(0.3)  # be polite

    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"Scraped {len(items)} assessments")

if __name__ == "__main__":
    scrape_catalog()
