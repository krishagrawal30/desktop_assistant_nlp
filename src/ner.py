import re
def extract_entities(text, intent):
    entities = {}
    file_pattern = r"([\w\-\_]+\.\w+)"

    if intent == "RENAME_FILE":
        pattern = r"([\w\-\_]+\.\w+)\s+(?:to|as)\s+([\w\-\_]+\.\w+)"
        match = re.search(pattern, text.lower())
        if match:
            entities["old_name"] = match.group(1)
            entities["new_name"] = match.group(2)

    elif intent == "MOVE_FILE":
        file_match = re.search(file_pattern, text.lower())
        folder_match = re.search(r"to\s+([\w\-\_]+)", text.lower())
        if file_match:
            entities["file"] = file_match.group(1)
        if folder_match:
            entities["destination"] = folder_match.group(1)

    elif intent == "DELETE_FILE":
        match = re.search(file_pattern, text.lower())
        if match:
            entities["file"] = match.group(1)

    elif intent == "OPEN_FILE":
        match = re.search(file_pattern, text.lower())
        if match:
            entities["file"] = match.group(1)
            
    elif intent == "RUN_FILE":
        match = re.search(file_pattern, text.lower())
        if match:
            entities["file"] = match.group(1)

    elif intent == "EXTRACT_ZIP":
        match = re.search(r"([\w\-\_]+\.zip)", text.lower())
        if match:
            entities["zip_file"] = match.group(1)

    elif intent == "WEB_SEARCH":
        query = re.sub(r"(search|google|look up|for)", "", text.lower())
        entities["query"] = query.strip()

    elif intent == "SEARCH_YOUTUBE":
        query = re.sub(r"(search youtube for|youtube|search)", "", text.lower())
        entities["query"] = query.strip()

    elif intent == "SEARCH_WIKIPEDIA":
        query = re.sub(r"(search wikipedia for|wikipedia|search)", "", text.lower())
        entities["query"] = query.strip()

    return entities