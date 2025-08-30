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
def parse_legal_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    articles = []

    for article_div in soup.find_all("div", class_="eli-subdivision", id=True):
        article_id = article_div["id"]
        if not article_id.startswith("art_"):
            continue
        article_number = article_id.split("_")[1]

        # Get article title
        title_tag = article_div.find("div", class_="eli-title")
        title_text = "No Title"
        if title_tag:
            p_tag = title_tag.find("p", class_="oj-sti-art")
            if p_tag:
                title_text = clean_text(p_tag.get_text())

        article_data = {
            "article_number": article_number,
            "title": title_text,
            "clauses": []
        }

        # Process each clause div
        for clause_div in article_div.find_all("div", id=True):
            clause_id = clause_div["id"]
            # Only process clause ids in {article}.{clause} format
            if re.match(r"\d{3}\.\d{3}", clause_id):
                clause_text_parts = []

                # 1. Group all <p> tags inside div
                for p_tag in clause_div.find_all("p", class_="oj-normal", recursive=False):
                    clause_text_parts.append(clean_text(p_tag.get_text()))

                # 2. Process tables inside div as subclauses
                for table in clause_div.find_all("table"):
                    for row in table.find_all("tr"):
                        cols = row.find_all("td")
                        if len(cols) >= 2:
                            label = clean_text(cols[0].get_text())
                            text = clean_text(cols[1].get_text())
                            subclause_text = f"{label} {text}" if label else text
                            clause_text_parts.append(subclause_text)

                full_clause_text = " ".join(clause_text_parts)

                article_data["clauses"].append({
                    "clause_id": clause_id,
                    "clause_text": full_clause_text
                })

        # Optionally: handle standalone <p class="oj-normal"> paragraphs that are not in any clause div
        for p_tag in article_div.find_all("p", class_="oj-normal"):
            parent_div = p_tag.find_parent("div", id=True)
            if parent_div is None or not re.match(r"\d{3}\.\d{3}", parent_div["id"]):
                text = clean_text(p_tag.get_text())
                if text:
                    article_data["clauses"].append({
                        "clause_id": f"{article_number}.standalone",
                        "clause_text": text
                    })

        articles.append(article_data)

    return articles

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
                if clause_div.get("id", "").startswith(f"{article_number}."):  # Matching clause ID pattern
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

            # Handle standalone clauses (paragraphs without divs with id)
            for p_tag in subdivision.find_all("p", class_="oj-normal"):
                standalone_clause_text = clean_text(p_tag.get_text(strip=True))
                if standalone_clause_text:  # Ensure it's not empty
                    article_data["clauses"].append({
                        "clause_id": f"{article_number}.standalone",  # Placeholder ID
                        "clause_text": standalone_clause_text
                    })

            # Add the article data to the articles list
            articles.append(article_data)
    
    # Save the articles list to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(articles, json_file, ensure_ascii=False, indent=4)

    print(f"Articles saved to {output_file}")

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

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def norm_num(s: str) -> str:
    # drop leading zeros for purely numeric strings
    return str(int(s)) if s.isdigit() else s

def parse_legal_html_to_json(html_path: str, output_path: str) -> None:
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    out = []

    # Only treat real articles: <div class="eli-subdivision" id="art_{number}">
    for art_div in soup.find_all("div", class_="eli-subdivision", id=True):
        art_id = art_div["id"]
        if not art_id.startswith("art_"):
            continue

        article_number = norm_num(art_id.split("_", 1)[1])

        # Article title (optional)
        title = "No Title"
        t = art_div.find("div", class_="eli-title")
        if t:
            p = t.find("p", class_="oj-sti-art")
            if p:
                title = clean_text(p.get_text(" "))

        article = {
            "article_number": article_number,
            "title": title,
            "clauses": []
        }

        # Clause divs are direct children with ids like 030.001
        clause_divs = art_div.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"), recursive=False)

        for cdiv in clause_divs:
            raw = cdiv["id"]               # e.g., "030.001"
            m = re.match(r"^(\d{3})\.(\d{3})$", raw)
            if not m:
                continue
            art3, cl3 = m.groups()
            art_norm = norm_num(art3)      # "30"
            cl_norm  = norm_num(cl3)       # "1"
            base_clause_id = f"{art_norm}.{cl_norm}"   # "30.1"

            # Collect immediate <p class="oj-normal"> in this clause (not from inside tables)
            immediate_ps = cdiv.find_all("p", class_="oj-normal", recursive=False)
            main_text_parts = [clean_text(p.get_text(" ")) for p in immediate_ps if clean_text(p.get_text(" "))]
            main_text = " ".join(main_text_parts)

            # Check if there are tables (subclauses)
            tables = cdiv.find_all("table", recursive=False)

            if tables:
                # If there’s main text (intro/conclusion), save it under 30.1
                if main_text:
                    article["clauses"].append({
                        "clause_id": base_clause_id,
                        "clause_text": main_text
                    })

                # Each table row -> subclause (label in first td, text in second td)
                for table in tables:
                    for row in table.find_all("tr"):
                        tds = row.find_all("td")
                        if len(tds) < 2:
                            continue
                        # label like "(a)" or "(1)"
                        label_raw = clean_text(tds[0].get_text(" "))
                        # normalize label to inner alnum only (a, 1, A, etc.)
                        label_inner = re.sub(r"[^A-Za-z0-9]", "", label_raw)
                        if not label_inner:
                            continue

                        # second column may have multiple <p>
                        right_ps = tds[1].find_all("p")
                        sub_text = " ".join(
                            clean_text(p.get_text(" ")) for p in right_ps if clean_text(p.get_text(" "))
                        ) or clean_text(tds[1].get_text(" "))

                        sub_clause_id = f"{art_norm}.{cl_norm}({label_inner})"  # e.g., "30.1(a)"
                        article["clauses"].append({
                            "clause_id": sub_clause_id,
                            "clause_text": sub_text
                        })
            else:
                # No tables: combine all immediate <p> into one chunk and save as 30.2, etc.
                if main_text:
                    article["clauses"].append({
                        "clause_id": base_clause_id,
                        "clause_text": main_text
                    })

        out.append(article)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(out)} articles to {output_path}")
    
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def norm_num(s: str) -> str:
    return str(int(s)) if s and s.isdigit() else s

def extract_table_right_text(table) -> str:
    """Get the text from the second <td> of each row, joining multiple <p>s."""
    parts = []
    for row in table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) >= 2:
            # Right cell may contain multiple <p>
            ps = tds[1].find_all("p")
            if ps:
                parts.extend(clean_text(p.get_text(" ")) for p in ps if clean_text(p.get_text(" ")))
            else:
                parts.append(clean_text(tds[1].get_text(" ")))
    return " ".join(p for p in parts if p)

def parse_legal_html_to_json(html_content: str) -> list[dict]:
    soup = BeautifulSoup(html_content, "html.parser")
    results = []

    # Iterate all legal subdivisions (articles & recitals)
    for sub in soup.find_all("div", class_="eli-subdivision", id=True):
        sid = sub["id"]

        # ---------- ARTICLES ----------
        if sid.startswith("art_"):
            article_number = norm_num(sid.split("_", 1)[1])

            # Title
            title = "No Title"
            t = sub.find("div", class_="eli-title")
            if t:
                pt = t.find("p", class_="oj-sti-art")
                if pt:
                    title = clean_text(pt.get_text(" "))

            article_obj = {
                "type": "article",
                "article_number": article_number,
                "title": title,
                "clauses": []
            }

            # Clause divs like 030.001
            for cdiv in sub.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"), recursive=False):
                raw = cdiv["id"]
                m = re.match(r"^(\d{3})\.(\d{3})$", raw)
                if not m:
                    continue
                art3, cl3 = m.groups()
                art_norm, cl_norm = norm_num(art3), norm_num(cl3)
                base_clause_id = f"{art_norm}.{cl_norm}"  # e.g., 30.1

                # Immediate <p> text (not from inside tables)
                ps = cdiv.find_all("p", class_="oj-normal", recursive=False)
                main_text = " ".join(
                    clean_text(p.get_text(" ")) for p in ps if clean_text(p.get_text(" "))
                )

                tables = cdiv.find_all("table", recursive=False)
                if tables:
                    # Keep preamble/conclusion as the base clause
                    if main_text:
                        article_obj["clauses"].append({
                            "clause_id": base_clause_id,
                            "clause_text": main_text
                        })
                    # Each table row -> subclause 30.1(a), 30.1(b), ...
                    for table in tables:
                        for row in table.find_all("tr"):
                            tds = row.find_all("td")
                            if len(tds) < 2:
                                continue
                            label_raw = clean_text(tds[0].get_text(" "))
                            label = re.sub(r"[^A-Za-z0-9]", "", label_raw)  # "(a)" -> "a"
                            if not label:
                                continue
                            right_text = extract_table_right_text(table) if tds[1].find("table") else None
                            sub_text = right_text or " ".join(
                                clean_text(p.get_text(" ")) for p in tds[1].find_all("p")
                            ) or clean_text(tds[1].get_text(" "))
                            article_obj["clauses"].append({
                                "clause_id": f"{base_clause_id}({label})",
                                "clause_text": sub_text
                            })
                else:
                    if main_text:
                        article_obj["clauses"].append({
                            "clause_id": base_clause_id,
                            "clause_text": main_text
                        })

            results.append(article_obj)
            continue

        # ---------- RECITALS (rct_###) ----------
        if sid.startswith("rct_"):
            rec_num_raw = sid.split("_", 1)[1]  # e.g., "119"
            rec_num = norm_num(rec_num_raw)

            # Recital text is usually in a single table:
            # left td -> "(119)" label; right td -> full text
            parts = []

            # Any standalone <p class="oj-normal"> directly under the recital (rare)
            for p in sub.find_all("p", class_="oj-normal", recursive=False):
                txt = clean_text(p.get_text(" "))
                if txt:
                    parts.append(txt)

            # Tables (common case)
            for table in sub.find_all("table", recursive=False):
                right_text = extract_table_right_text(table)
                if right_text:
                    parts.append(right_text)

            recital_text = clean_text(" ".join(parts))

            recital_obj = {
                "type": "recital",
                "article_number": "R",                 # keep pipeline happy; distinguishes from numeric articles
                "recital_number": rec_num,             # convenience
                "title": f"Recital {rec_num}",
                "clauses": [
                    {
                        "clause_id": f"R{rec_num}",
                        "clause_text": recital_text
                    }
                ]
            }
            results.append(recital_obj)

    return results

# ---------- Example save helper ----------
def save_parsed_to_json(html_file: str, output_json: str):
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()
    data = parse_legal_html_to_json(html)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} entries (articles + recitals) to {output_json}")
    
save_parsed_to_json("./files/EU_DSA.html", "articles.json")