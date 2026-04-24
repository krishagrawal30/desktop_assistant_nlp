import os
import shutil
import zipfile
from urllib.parse import quote, quote_plus


SEARCH_ROOT = "C:\\Users\\agraw"

SYSTEM_COMMANDS = {"SHUTDOWN", "RESTART", "SUSPEND"}
SUPPORTED_INTENTS = {
    "RENAME_FILE",
    "MOVE_FILE",
    "COPY_FILE",
    "CREATE_FILE",
    "DELETE_FILE",
    "SEARCH_FILE",
    "OPEN_FILE",
    "RUN_FILE",
    "EXTRACT_ZIP",
    "SELECT_MULTIPLE",
    "WEB_SEARCH",
    "SEARCH_YOUTUBE",
    "SEARCH_WIKIPEDIA",
    "SHUTDOWN",
    "RESTART",
    "SUSPEND",
}
REQUIRED_ENTITY_KEYS = {
    "RENAME_FILE": ["old_name", "new_name"],
    "MOVE_FILE": ["file", "destination"],
    "COPY_FILE": ["file", "destination"],
    "CREATE_FILE": ["file"],
    "DELETE_FILE": ["file"],
    "SEARCH_FILE": ["file"],
    "OPEN_FILE": ["file"],
    "RUN_FILE": ["file"],
    "EXTRACT_ZIP": ["zip_file"],
    "WEB_SEARCH": ["query"],
    "SEARCH_YOUTUBE": ["query"],
    "SEARCH_WIKIPEDIA": ["query"],
}


def _result(intent, status, message, dry_run=False, details=None):
    payload = {
        "intent": intent,
        "status": status,
        "message": message,
        "dry_run": dry_run,
    }
    if details is not None:
        payload["details"] = details
    return payload


def _missing_required(intent, entities):
    required = REQUIRED_ENTITY_KEYS.get(intent, [])
    return [key for key in required if not entities.get(key)]


def _select_multiple_missing(entities):
    files = entities.get("files") or []
    extension = entities.get("extension")
    return not files and not extension


def find_folder(folder_name, search_path=None):
    if search_path is None:
        search_path = os.path.expanduser("~")

    for root, dirs, _ in os.walk(search_path):
        for folder in dirs:
            if folder_name.lower() in folder.lower():
                return os.path.join(root, folder)

    return None


def find_file(filename, search_path=SEARCH_ROOT):
    target = filename.lower()
    for root, _, files in os.walk(search_path):
        for current_name in files:
            if current_name.lower() == target:
                return os.path.join(root, current_name)
    return None


def find_multiple(extension, search_path=SEARCH_ROOT):
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"

    results = []
    for root, _, files in os.walk(search_path):
        for current_name in files:
            if current_name.lower().endswith(normalized):
                results.append(os.path.join(root, current_name))
    return results


def open_with_system(target):
    if target.startswith("http"):
        os.startfile(target)
    else:
        target = os.path.abspath(target)
        target = os.path.normpath(target)
        print("Opening:", target)
        os.startfile(target)


def execute(intent, entities, dry_run=False):
    entities = entities or {}

    if dry_run:
        if intent not in SUPPORTED_INTENTS:
            return _result(intent, "not_implemented", "Intent route is not implemented", dry_run=True)

        if intent == "SELECT_MULTIPLE":
            if _select_multiple_missing(entities):
                return _result(
                    intent,
                    "missing_entities",
                    "SELECT_MULTIPLE requires either 'files' or 'extension'",
                    dry_run=True,
                )
            return _result(intent, "success", "Dry run validated SELECT_MULTIPLE route", dry_run=True)

        missing = _missing_required(intent, entities)
        if missing:
            return _result(
                intent,
                "missing_entities",
                f"Missing required entities: {', '.join(missing)}",
                dry_run=True,
            )

        return _result(intent, "success", "Dry run validation passed", dry_run=True)

    try:
        if intent == "RENAME_FILE":
            old_name = entities.get("old_name")
            new_name = entities.get("new_name")
            if not old_name or not new_name:
                print("Missing file names for rename")
                return _result(intent, "missing_entities", "Missing old_name/new_name")

            file_path = find_file(old_name)
            if not file_path:
                print("File not found")
                return _result(intent, "not_found", "File not found", details={"file": old_name})

            new_path = os.path.join(os.path.dirname(file_path), new_name)
            os.rename(file_path, new_path)
            print("File renamed successfully")
            return _result(intent, "success", "File renamed successfully", details={"new_path": new_path})

        if intent == "MOVE_FILE":
            file_name = entities.get("file")
            destination = entities.get("destination")
            if not file_name or not destination:
                print("Missing file or destination")
                return _result(intent, "missing_entities", "Missing file/destination")

            file_path = find_file(file_name)
            folder_path = find_folder(destination)
            if not file_path or not folder_path:
                print("File or destination folder not found")
                return _result(intent, "not_found", "File or destination folder not found")

            new_path = os.path.join(folder_path, os.path.basename(file_path))
            shutil.move(file_path, new_path)
            print("File moved successfully")
            return _result(intent, "success", "File moved successfully", details={"new_path": new_path})

        if intent == "COPY_FILE":
            file_name = entities.get("file")
            destination = entities.get("destination")
            if not file_name or not destination:
                print("Missing file or destination")
                return _result(intent, "missing_entities", "Missing file/destination")

            file_path = find_file(file_name)
            folder_path = find_folder(destination)
            if not file_path or not folder_path:
                print("File or destination folder not found")
                return _result(intent, "not_found", "File or destination folder not found")

            new_path = os.path.join(folder_path, os.path.basename(file_path))
            shutil.copy(file_path, new_path)
            print("File copied successfully")
            return _result(intent, "success", "File copied successfully", details={"new_path": new_path})

        if intent == "CREATE_FILE":
            name = entities.get("file")
            if not name:
                print("Missing file name")
                return _result(intent, "missing_entities", "Missing file")

            path = os.path.join(SEARCH_ROOT, name)
            with open(path, "w", encoding="utf-8"):
                pass
            print("File created:", path)
            return _result(intent, "success", "File created", details={"path": path})

        if intent == "DELETE_FILE":
            file_name = entities.get("file")
            if not file_name:
                print("Missing file name")
                return _result(intent, "missing_entities", "Missing file")

            file_path = find_file(file_name)
            if not file_path:
                print("File not found")
                return _result(intent, "not_found", "File not found", details={"file": file_name})

            os.remove(file_path)
            print("File deleted")
            return _result(intent, "success", "File deleted", details={"path": file_path})

        if intent == "SEARCH_FILE":
            file_name = entities.get("file")
            if not file_name:
                print("Missing file name")
                return _result(intent, "missing_entities", "Missing file")

            file_path = find_file(file_name)
            if file_path:
                print("File found at:", file_path)
                return _result(intent, "success", "File found", details={"path": file_path})

            print("File not found")
            return _result(intent, "not_found", "File not found", details={"file": file_name})

        if intent == "OPEN_FILE":
            file_name = entities.get("file")
            if not file_name:
                print("Missing file name")
                return _result(intent, "missing_entities", "Missing file")

            file_path = find_file(file_name)
            if not file_path:
                print("File not found")
                return _result(intent, "not_found", "File not found", details={"file": file_name})

            open_with_system(file_path)
            return _result(intent, "success", "File opened", details={"path": file_path})

        if intent == "RUN_FILE":
            file_name = entities.get("file")
            if not file_name:
                print("Missing file name")
                return _result(intent, "missing_entities", "Missing file")

            file_path = find_file(file_name)
            if not file_path:
                print("File not found")
                return _result(intent, "not_found", "File not found", details={"file": file_name})

            open_with_system(file_path)
            return _result(intent, "success", "File executed", details={"path": file_path})

        if intent == "EXTRACT_ZIP":
            zip_file = entities.get("zip_file")
            if not zip_file:
                print("Missing zip file name")
                return _result(intent, "missing_entities", "Missing zip_file")

            file_path = find_file(zip_file)
            if not file_path:
                print("Zip file not found")
                return _result(intent, "not_found", "Zip file not found", details={"file": zip_file})

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(file_path))
            print("Zip extracted")
            return _result(intent, "success", "Zip extracted", details={"path": file_path})

        if intent == "SELECT_MULTIPLE":
            files = entities.get("files") or []
            extension = entities.get("extension")

            if files:
                found_files = [path for path in (find_file(name) for name in files) if path]
                if found_files:
                    print("Files selected:")
                    for found in found_files:
                        print(found)
                    return _result(intent, "success", "Multiple files selected", details={"count": len(found_files)})

                print("No matching files found")
                return _result(intent, "not_found", "No matching files found")

            if extension:
                matched = find_multiple(extension)
                if matched:
                    print(f"Found {len(matched)} files with extension {extension}")
                    return _result(intent, "success", "Files selected by extension", details={"count": len(matched)})

                print("No files found for extension")
                return _result(intent, "not_found", "No files found for extension")

            print("No files or extension provided")
            return _result(intent, "missing_entities", "Missing files/extension")

        if intent == "WEB_SEARCH":
            query = entities.get("query")
            if not query:
                print("Missing query")
                return _result(intent, "missing_entities", "Missing query")

            url = "https://www.google.com/search?q=" + quote_plus(query)
            open_with_system(url)
            return _result(intent, "success", "Web search opened", details={"url": url})

        if intent == "SEARCH_YOUTUBE":
            query = entities.get("query")
            if not query:
                print("Missing query")
                return _result(intent, "missing_entities", "Missing query")

            url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
            open_with_system(url)
            return _result(intent, "success", "YouTube search opened", details={"url": url})

        if intent == "SEARCH_WIKIPEDIA":
            query = entities.get("query")
            if not query:
                print("Missing query")
                return _result(intent, "missing_entities", "Missing query")

            url = "https://en.wikipedia.org/wiki/" + quote(query.replace(" ", "_"))
            open_with_system(url)
            return _result(intent, "success", "Wikipedia page opened", details={"url": url})

        if intent == "SHUTDOWN":
            return_code = os.system("shutdown /s /t 1")
            status = "success" if return_code == 0 else "failure"
            return _result(intent, status, "Shutdown command executed")

        if intent == "RESTART":
            return_code = os.system("shutdown /r /t 1")
            status = "success" if return_code == 0 else "failure"
            return _result(intent, status, "Restart command executed")

        if intent == "SUSPEND":
            return_code = os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            status = "success" if return_code == 0 else "failure"
            return _result(intent, status, "Suspend command executed")

        print("Command not recognized")
        return _result(intent, "not_implemented", "Command not recognized")
    except Exception as exc:
        print(f"Execution failed: {exc}")
        return _result(intent, "failure", str(exc))