import re

def extract_entities(text,intent):

    text=text.lower()
    entities={}

    file_pattern=r"([\w\-\_]+\.[a-z0-9]+)"

    if intent=="RENAME_FILE":
        match=re.search(rf"{file_pattern}\s+(?:to|as)\s+{file_pattern}",text)
        if match:
            entities["old_name"]=match.group(1)
            entities["new_name"]=match.group(2)

    elif intent in ["MOVE_FILE","COPY_FILE"]:
        file_match=re.search(file_pattern,text)
        dest_match=re.search(r"(?:to|into)\s+([\w\-\_\\/:]+)",text)
        if file_match:
            entities["file"]=file_match.group(1)
        if dest_match:
            entities["destination"]=dest_match.group(1)

    elif intent in ["DELETE_FILE","OPEN_FILE","RUN_FILE","SEARCH_FILE"]:
        match=re.search(file_pattern,text)
        if match:
            entities["file"]=match.group(1)

    elif intent=="CREATE_FILE":
        match=re.search(file_pattern,text)
        if match:
            entities["file"]=match.group(1)

    elif intent=="SELECT_MULTIPLE":
        match=re.search(r"(?:all|multiple)\s+(\w+)",text)
        if match:
            entities["extension"]="."+match.group(1)

    elif intent=="EXTRACT_ZIP":
        match=re.search(r"([\w\-\_]+\.zip)",text)
        if match:
            entities["zip_file"]=match.group(1)

    elif intent == "WEB_SEARCH":
        match = re.search(
            r"(?:search\s+for|search|find|look\s+up)\s+(.*?)\s+(?:on\s+)?(?:google|web)",
            text.lower()
        )
        if match:
            entities["query"] = match.group(1).strip()

    elif intent == "SEARCH_YOUTUBE":
        match = re.search(
            r"(?:search\s+for|search|find|look\s+up)\s+(.*?)\s+(?:on\s+)?youtube",
            text.lower()
        )
        if match:
            entities["query"] = match.group(1).strip()

    elif intent == "SEARCH_WIKIPEDIA":
        match = re.search(
            r"(?:search\s+for|search|find|look\s+up)\s+(.*?)\s+(?:on\s+)?wikipedia",
            text.lower()
        )
        if match:
            entities["query"] = match.group(1).strip()

    return entities