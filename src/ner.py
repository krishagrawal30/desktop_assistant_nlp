import re

FILE_PATTERN = r"([\w\-_]+\.[a-z0-9]+)"
ZIP_PATTERN = r"([\w\-_.]+\.zip)"


def _clean_destination(destination):
    cleaned = re.sub(r"\b(folder|directory)\b", "", destination).strip()
    return re.sub(r"\s+", " ", cleaned).strip(" .")


def _extract_generic_query(text):
    match = re.search(r"(?:search(?:\s+for)?|find|look\s+for|look\s+up|google)\s+(.+)", text)
    if not match:
        return None

    query = match.group(1).strip()
    query = re.sub(r"^(for\s+)", "", query)
    query = re.sub(r"\s+(?:on\s+)?(?:google|web)$", "", query).strip()
    return query or None


def extract_entities(text, intent):

    text = text.lower()
    entities = {}

    if intent == "RENAME_FILE":
        match = re.search(rf"{FILE_PATTERN}\s+(?:to|as|into)\s+{FILE_PATTERN}", text)
        if match:
            entities["old_name"] = match.group(1)
            entities["new_name"] = match.group(2)

    elif intent in ["MOVE_FILE", "COPY_FILE"]:
        file_match = re.search(FILE_PATTERN, text)
        dest_match = re.search(
            r"(?:to|into|inside|in)\s+(?:the\s+)?([\w\-_\\/: ]+?)(?:\s*(?:folder|directory))?$",
            text,
        )

        if file_match:
            entities["file"] = file_match.group(1)

        if dest_match:
            cleaned_destination = _clean_destination(dest_match.group(1))
            if cleaned_destination:
                entities["destination"] = cleaned_destination

    elif intent in ["DELETE_FILE", "OPEN_FILE", "RUN_FILE", "SEARCH_FILE"]:
        match = re.search(FILE_PATTERN, text)
        if match:
            entities["file"] = match.group(1)

    elif intent == "CREATE_FILE":
        match = re.search(FILE_PATTERN, text)
        if match:
            entities["file"] = match.group(1)

    elif intent == "SELECT_MULTIPLE":
        files = re.findall(FILE_PATTERN, text)
        if len(files) >= 2:
            entities["files"] = files
        else:
            extension_match = re.search(r"(?:all|multiple)\s+\.?([a-z0-9]+)\s+files?", text)
            if extension_match:
                entities["extension"] = "." + extension_match.group(1)

    elif intent == "EXTRACT_ZIP":
        match = re.search(ZIP_PATTERN, text)
        if match:
            entities["zip_file"] = match.group(1)

    elif intent == "WEB_SEARCH":
        query = _extract_generic_query(text)
        if query:
            entities["query"] = query

    elif intent == "SEARCH_YOUTUBE":
        match = re.search(r"search\s+youtube\s+for\s+(.+)", text)
        if not match:
            match = re.search(r"(?:search\s+for|search|find|look\s+for|look\s+up)\s+(.*?)\s+(?:on\s+)?youtube", text)
        if match:
            entities["query"] = match.group(1).strip()

    elif intent == "SEARCH_WIKIPEDIA":
        match = re.search(r"search\s+wikipedia\s+for\s+(.+)", text)
        if not match:
            match = re.search(r"(?:search\s+for|search|find|look\s+for|look\s+up)\s+(.*?)\s+(?:on\s+)?wikipedia", text)
        if match:
            entities["query"] = match.group(1).strip()

    return entities