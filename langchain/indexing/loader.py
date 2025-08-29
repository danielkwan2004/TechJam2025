import json
from bs4 import BeautifulSoup
import re

# Step 1: Read HTML File
def read_html_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    print(f"HTML file loaded from {file_path}")
    return html_content

# Step 2: Parse HTML and Chunk by Article, Section, and Clause
def parse_html_and_chunk(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    clauses = []

    # Iterate over all subdivisions (e.g., articles) with 'id' attribute
    for subdivision in soup.find_all("div", id=True):
        clause_id = subdivision.get("id", "")
        
        # Check if the div is an article (id starts with 'art_')
        if clause_id.startswith("art_"):
            article_number = clause_id.split("_")[-1]  # Extract article number from 'art_{number}'
            
            # Extract title of the article
            title_tag = subdivision.find("div", class_="eli-title")
            title = "No Title"
            if title_tag:
                title_p_tag = title_tag.find("p", class_="oj-sti-art")
                if title_p_tag:
                    title = title_p_tag.get_text(strip=True)

            # Process each clause (identified by ids like 040.001, 040.002, etc.)
            for clause_div in subdivision.find_all("div", id=True):
                clause_text = ""
                if clause_div.get("id", "").startswith(f"{article_number}."):  # Matching clause ID pattern (e.g., 040.001)
                    for p_tag in clause_div.find_all("p", class_="oj-normal"):
                        clause_text += p_tag.get_text(" ", strip=True)  # Collect clause text

                    # Extract sub-regulations (inside tables)
                    sub_regulations = []
                    for table in clause_div.find_all("table"):
                        for row in table.find_all("tr"):
                            cols = row.find_all("td")
                            if len(cols) > 1:
                                sub_regulations.append(cols[1].get_text(strip=True))

                    # Combine clause text and sub-regulations
                    full_clause_text = clause_text + "\n" + "\n".join(sub_regulations)

                    # Store the chunk (clause text + metadata)
                    clause_metadata = {
                        "article_number": article_number,
                        "title": title,
                        "clause_id": clause_div.get("id", ""),
                        "clause_text": full_clause_text
                    }
                    clauses.append(clause_metadata)
    
    return clauses

def save_articles_as_json(html_content, output_file="articles.json"):
    soup = BeautifulSoup(html_content, "html.parser")
    articles = []

    # Iterate over all subdivisions (e.g., articles) with 'class' and 'id' attributes
    for subdivision in soup.find_all("div", class_="eli-subdivision", id=True):
        clause_id = subdivision.get("id", "")
        
        # Only process divs that have class="eli-subdivision" and id starting with 'art_'
        if clause_id.startswith("art_"):
            article_number = clause_id.split("_")[-1]  # Extract article number from 'art_{number}'
            
            # Extract article title
            title_tag = subdivision.find("div", class_="eli-title")
            title = "No Title"
            if title_tag:
                title_p_tag = title_tag.find("p", class_="oj-sti-art")
                if title_p_tag:
                    title = clean_text(title_p_tag.get_text(strip=True))  # Clean title text
            
            # Prepare article data
            article_data = {
                "article_number": article_number,
                "title": title,
                "clauses": []
            }

            # Process each clause under the article (identified by ids like 040.001, 040.002, etc.)
            for clause_div in subdivision.find_all("div", id=True):
                clause_text = ""
                padded_article_number = article_number.zfill(3)
                if clause_div.get("id", "").startswith(f"{padded_article_number}."):  # Matching clause ID pattern
                    for p_tag in clause_div.find_all("p", class_="oj-normal"):
                        clause_text += clean_text(p_tag.get_text(" ", strip=True))  # Clean clause text
                    
                    # Extract sub-regulations (inside tables)
                    sub_regulations = []
                    for table in clause_div.find_all("table"):
                        for row in table.find_all("tr"):
                            cols = row.find_all("td")
                            if len(cols) > 1:
                                sub_regulations.append(clean_text(cols[1].get_text(strip=True)))  # Clean sub-regulation text

                    # Combine clause text and sub-regulations
                    full_clause_text = clause_text + "\n" + "\n".join(sub_regulations)

                    # Append clause data to the article
                    article_data["clauses"].append({
                        "clause_id": clause_div.get("id", ""),
                        "clause_text": full_clause_text
                    })

            # Add the article data to the articles list
            articles.append(article_data)
    
    # Save the articles list to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(articles, json_file, ensure_ascii=False, indent=4)

    print(f"Articles saved to {output_file}")

# Main execution


# Step 3: Save Output to JSON File
def save_output_to_json(clauses, output_file="parsed_output.json"):
    output_data = []
    
    for (clause, metadata) in clauses:
        data = {
            "clause_text": clause,
            "metadata": metadata
        }
        output_data.append(data)
    
    # Save the output to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Output saved to {output_file}")

# Main Execution
file_path = './files/EU_DSA.html'  # Make sure this file exists at the specified path
html_content = read_html_from_file(file_path)

def clean_text(text):
    # Remove extra newlines and multiple spaces
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace sequences with a single space
    text = text.strip()  # Remove leading and trailing whitespace
    return text

html_content = clean_text(html_content)
save_articles_as_json(html_content, "articles.json")