import os
import shutil
import subprocess
import webbrowser
import zipfile

def execute(intent, entities):
    if intent == "RENAME_FILE":
        old_name = entities.get("old_name")
        new_name = entities.get("new_name")
        if old_name and new_name:
            os.rename(old_name, new_name)
            print("File renamed successfully")

    elif intent == "MOVE_FILE":
        file = entities.get("file")
        destination = entities.get("destination")
        if file and destination:
            shutil.move(file, destination)
            print("File moved successfully")

    elif intent == "DELETE_FILE":
        file = entities.get("file")
        if file:
            os.remove(file)
            print("File deleted")

    elif intent == "OPEN_FILE":
        file = entities.get("file")
        if file:
            subprocess.run(["xdg-open", file])

    elif intent == "RUN_FILE":
        file = entities.get("file")
        if file:
            subprocess.run(["xdg-open", file])

    elif intent == "EXTRACT_ZIP":
        zip_file = entities.get("zip_file")
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall()
            print("Zip extracted")

    elif intent == "WEB_SEARCH":
        query = entities.get("query")
        if query:
            webbrowser.open(
                "https://www.google.com/search?q=" + query.replace(" ", "+")
            )

    elif intent == "SEARCH_YOUTUBE":
        query = entities.get("query")
        if query:
            webbrowser.open(
                "https://www.youtube.com/results?search_query=" + query.replace(" ", "+")
            )

    elif intent == "SEARCH_WIKIPEDIA":
        query = entities.get("query")
        if query:
            webbrowser.open(
                "https://en.wikipedia.org/wiki/" + query.replace(" ", "_")
            )