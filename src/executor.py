import os
import shutil
import zipfile

SEARCH_ROOT="C:\\Users\\agraw\\OneDrive\\Desktop"

def find_file(filename,search_path=SEARCH_ROOT):
    for root,dirs,files in os.walk(search_path):
        if filename in files:
            return os.path.join(root,filename)
    return None

def open_with_system(target):
    target = os.path.abspath(target)
    target = os.path.normpath(target)
    print("Opening:", target)
    os.startfile(target)

def execute(intent,entities):

    if intent=="RENAME_FILE":
        old_name=entities.get("old_name")
        new_name=entities.get("new_name")
        file_path=find_file(old_name)
        if file_path:
            new_path=os.path.join(os.path.dirname(file_path),new_name)
            os.rename(file_path,new_path)
            print("File renamed successfully")
        else:
            print("File not found")

    elif intent=="MOVE_FILE":
        file=entities.get("file")
        destination=entities.get("destination")
        file_path=find_file(file)
        if file_path:
            shutil.move(file_path,destination)
            print("File moved successfully")
        else:
            print("File not found")

    elif intent=="DELETE_FILE":
        file=entities.get("file")
        file_path=find_file(file)
        if file_path:
            os.remove(file_path)
            print("File deleted")
        else:
            print("File not found")

    elif intent=="OPEN_FILE":
        file=entities.get("file")
        file_path=find_file(file)
        if file_path:
            open_with_system(file_path)
        else:
            print("File not found")

    elif intent=="RUN_FILE":
        file=entities.get("file")
        file_path=find_file(file)
        if file_path:
            open_with_system(file_path)
        else:
            print("File not found")

    elif intent=="EXTRACT_ZIP":
        zip_file=entities.get("zip_file")
        file_path=find_file(zip_file)
        if file_path:
            with zipfile.ZipFile(file_path,'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(file_path))
            print("Zip extracted")
        else:
            print("Zip file not found")

    elif intent=="WEB_SEARCH":
        query=entities.get("query")
        if query:
            url="https://www.google.com/search?q="+query.replace(" ","+")
            open_with_system(url)

    elif intent=="SEARCH_YOUTUBE":
        query=entities.get("query")
        if query:
            url="https://www.youtube.com/results?search_query="+query.replace(" ","+")
            open_with_system(url)

    elif intent=="SEARCH_WIKIPEDIA":
        query=entities.get("query")
        if query:
            url="https://en.wikipedia.org/wiki/"+query.replace(" ","_")
            open_with_system(url)