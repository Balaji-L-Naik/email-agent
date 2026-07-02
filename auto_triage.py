import typer
from rich.console import Console
from rich.panel import Panel
from langchain_core.messages import HumanMessage
from agent.graph import graph

app = typer.Typer(help="Run your daily email triage automatically.")
console = Console()

@app.command()
def run(
    keywords: str = typer.Argument(..., help="Comma-separated keywords (e.g., 'offers, remote')"),
    label: str = typer.Option(None, "--label", "-l", help="Label to apply to matched emails (e.g., 'IMPORTANT')"),
    tasks: bool = typer.Option(False, "--tasks", "-t", help="Extract deadlines into Google Tasks"),
    timeframe: str = typer.Option("1d", "--time", help="Timeframe to search (e.g., '1d' for today, '7d' for this week')"),
    max_results: int = typer.Option(20, "--max", help="Max number of emails to process"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Use local AI Vector Database for conceptual meaning")
):
    console.print("[bold cyan]🚀 Initializing Automated Daily Triage...[/bold cyan]")
    console.print(f"🔍 [bold]Keywords:[/bold] {keywords}")
    console.print(f"📅 [bold]Timeframe:[/bold] {timeframe} | 🔢 [bold]Max Emails:[/bold] {max_results}")
    if label:
        console.print(f"🏷️  [bold]Label:[/bold] {label}")
    if tasks:
        console.print(f"📝 [bold]Tasks:[/bold] Enabled")
    if semantic:
        console.print(f"🧠 [bold]Search Mode:[/bold] Semantic (Vector DB)")
        
    keyword_list = [k.strip() for k in keywords.split(",")]

    # Directly inject perfectly formatted config to bypass the LLM chat-parsing bottleneck
    state = {
        "messages": [HumanMessage(content="Automated Run Triggered")],
        "emails": [],
        "pending_send": None,
        "pending_delete": None,
        "pending_todo": None,
        "triage_config": {
            "keywords": keyword_list,
            "add_tasks": tasks,
            "label_to_add": label,
            "timeframe": timeframe,
            "max_results": max_results,
            "use_semantic": semantic
        },
        "intent": "triage", 
        "response": ""
    }

    with console.status("[cyan]🧠 Running workflow... (Fetching, reading, summarizing, and emailing)[/cyan]", spinner="dots"):
        final_event = None
        try:
            for event in graph.stream(state):
                final_event = event
        except Exception as e:
            console.print(f"\n[bold red]⚠ Error during triage:[/bold red] {e}")
            raise typer.Exit(1)

    # Extract and print the final response
    if final_event:
        node_name = list(final_event.keys())[0]
        response_text = final_event[node_name].get("response", "Done.")
        
        console.print("\n[bold green]✅ Job Complete![/bold green]")
        console.print(Panel(response_text, title="[bold cyan]Execution Summary[/bold cyan]", border_style="cyan"))

if __name__ == "__main__":
    app()