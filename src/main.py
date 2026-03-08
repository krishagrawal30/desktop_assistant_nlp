import joblib
from ner import extract_entities
from executor import execute
from rich.console import Console
from rich.panel import Panel

console = Console()

model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

console.print(Panel.fit("Desktop Assistant ", style="bold cyan"))
console.print("[dim]Type 'exit' to quit[/dim]\n")

while True:

    text = console.input("[bold green]Enter command:[/bold green] ")

    if text.lower() in ["exit", "quit"]:
        console.print("[bold red]Exiting assistant...[/bold red]")
        break

    X = vectorizer.transform([text])
    intent = model.predict(X)[0]

    console.print(f"[yellow]Predicted Intent:[/yellow] {intent}")

    entities = extract_entities(text, intent)

    console.print(f"[magenta]Extracted Entities:[/magenta] {entities}")

    execute(intent, entities)