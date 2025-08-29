from bs4 import BeautifulSoup

# Sample HTML content for testing
html_content = """
<div class="eli-subdivision" id="art_2">
    <p id="d1e1644-1-1" class="oj-ti-art">Article 2</p>
    <div class="eli-title" id="art_2.tit_1">
        <p class="oj-sti-art">Scope</p>
    </div>
    <div id="002.001">
        <p class="oj-normal">1. This Regulation shall apply...</p>
    </div>
    <div id="002.002">
        <p class="oj-normal">2. This Regulation shall not apply...</p>
    </div>
</div>
"""

# Parse the HTML to see the structure
soup = BeautifulSoup(html_content, "html.parser")

subdivisions = soup.find_all("div", class_="eli-subdivision")
print(f"Found {len(subdivisions)} subdivisions")
for subdivision in subdivisions:
    print(subdivision)  # To verify content inside each subdivision

# Check for specific article p tags
articles = soup.find_all("p", class_="oj-ti-art")
print(f"Found {len(articles)} article tags")
for article in articles:
    print(article.get_text())  # Print the text inside each article