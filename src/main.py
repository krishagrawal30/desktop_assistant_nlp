import joblib
from ner import extract_entities
from executor import execute
from rich.console import Console
from rich.panel import Panel

console = Console()

model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

console.print(Panel.fit("Desktop Assistant NLP", style="bold cyan"))
console.print("Type 'exit' to quit\n")

while True:

    text = console.input("[bold green]Enter command:[/bold green] ")

    if text.lower() in ["exit","quit"]:
        console.print("[bold red]Exiting assistant...[/bold red]")
        break

    X = vectorizer.transform([text])
    intent = model.predict(X)[0]

    entities = extract_entities(text, intent)

    # commands that do not require entities
    SYSTEM_COMMANDS = {"SHUTDOWN","RESTART","SUSPEND"}

    # if intent needs entities but none extracted → reject command
    if intent not in SYSTEM_COMMANDS and not entities:
        console.print("[red]Command not recognized.[/red]")
        continue

    console.print(f"[yellow]Predicted Intent:[/yellow] {intent}")
    console.print(f"[magenta]Extracted Entities:[/magenta] {entities}")

    execute(intent, entities)