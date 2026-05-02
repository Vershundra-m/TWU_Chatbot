import requests
from bs4 import BeautifulSoup
import config
import re

def get_clean_text():
    """Scrape and clean text from TWU webpage - captures ALL content"""
    
    # Get the webpage
    response = requests.get(config.INFORMATICS_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove only pure code elements (not content)
    if soup.head:
        soup.head.decompose()
    
    for tag in soup.find_all(['script', 'style', 'noscript']):
        tag.decompose()
    
    # Collect text from ALL important sections
    text_parts = []
    
    # Main content 
    main_content = soup.find('main')
    if main_content:
        text_parts.append(main_content.get_text())
    
    # Sidebar 
    sidebar = soup.find('div', class_='sidebar')
    if sidebar:
        text_parts.append(sidebar.get_text())
        
    # Accordion sections
    accordions = soup.find_all('div', {'data-aria-accordion': True})
    for accordion in accordions:
        text_parts.append(accordion.get_text())
    
    # Jump-scroll divs (contain deadlines and contact)
    jump_scrolls = soup.find_all('div', class_='jump-scroll')
    for js in jump_scrolls:
        text_parts.append(js.get_text())
    
    # Feature boxes 
    feature_boxes = soup.find_all('div', class_='feature')
    for feature in feature_boxes:
        text_parts.append(feature.get_text())
    
    
    # Combine all parts
    raw_text = '\n'.join(text_parts)
    
    # Clean up the text 
    lines = []
    for line in raw_text.split('\n'):
        line = line.strip()
        # Keep lines that have meaningful content
        if line and len(line) > 2:
            lines.append(line)
    
    return '\n'.join(lines)

def chunk_text(text):
    """Split text into chunks"""
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=lambda x: len(tokenizer.encode(x)),
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return splitter.split_text(text)
