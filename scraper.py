import requests
from bs4 import BeautifulSoup
import config

def get_clean_text():
    """Scrape and clean text from TWU webpage"""
    
    # Get the webpage
    response = requests.get(config.INFORMATICS_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove head, scripts, styles
    if soup.head:
        soup.head.decompose()
    
    for tag in soup.find_all(['script', 'style', 'noscript']):
        tag.decompose()
    
    # Find main content
    main_content = soup.find('main')
    if main_content:
        raw_text = main_content.get_text()
    else:
        raw_text = soup.get_text()
    
    # Clean up the text
    lines = []
    for line in raw_text.split('\n'):
        line = line.strip()
        if line and len(line) > 15:
            skip_words = ['skip to main', 'students', 'faculty & staff', 'alumni', 'visitors', 
                          'parents', 'denton', 'dallas', 'houston', 'libraries', 'a-z index', 
                          'directories', 'search', 'menu', "texas woman's university",
                          'about', 'maps', 'parking', 'contact twu', 'calendar', 'administration',
                          'attractions', 'traditions', 'history', 'academics', 'admissions',
                          'student life', 'research', 'athletics', 'giving', 'apply', 'visit us']
            
            skip = False
            for skip_word in skip_words:
                if skip_word in line.lower():
                    skip = True
                    break
            
            if not skip:
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
