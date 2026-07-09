"""
auto_triage_file.py
-------------------
Cron-friendly daily triage script. Reads keywords from a .txt file,
runs vector sync to keep the semantic DB current, then executes the
full triage workflow (search, summarize, label, extract tasks, email digest).

File format (one keyword per line; # for comments):
    # Job applications
    internship
    application
    offer

Usage:
    python auto_triage_file.py --file keywords.txt [FLAGS]

Flags (all optional):
    --label, -l      Label to apply (e.g. 'IMPORTANT')
    --tasks, -t      Extract deadlines into Google Tasks
    --time            Search timeframe (default: 1d)
    --max             Max emails to process (default: 20)
    --semantic, -s   Use vector DB for conceptual search
    --skip-sync       Skip the vector database sync at startup
"""

import typer
import sys
import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from langchain_core.messages import HumanMessage
from agent.graph import graph

# Ensure project root is on sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.vector_sync import sync_recent_emails

app = typer.Typer(help="Run daily triage using keywords from a .txt file.")
console = Console()


@app.command()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Path to a .txt file with keywords (one per line, # for comments)"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Label to apply to matched emails (e.g., 'IMPORTANT')"),
    tasks: bool = typer.Option(False, "--tasks", "-t", help="Extract deadlines into Google Tasks"),
    timeframe: str = typer.Option("1d", "--time", help="Timeframe to search (e.g., '1d' for today, '7d' for this week')"),
    max_results: int = typer.Option(20, "--max", help="Max number of emails to process"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Use local AI Vector Database for conceptual meaning"),
    skip_sync: bool = typer.Option(False, "--skip-sync", help="Skip the vector database sync at startup"),
):
    console.print("[bold cyan]🚀 Starting Daily Triage (from file)...[/bold cyan]")

    # --- 1. Read keywords file ---
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[bold red]✖ File not found:[/bold red] {file}")
        raise typer.Exit(1)

    keywords = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keywords.append(line)

    if not keywords:
        console.print("[bold red]✖ No keywords found in the file.[/bold red]")
        raise typer.Exit(1)

    keyword_str = ", ".join(keywords)

    # --- 2. Vector sync (keeps the semantic DB fresh for both keyword and semantic modes) ---
    if not skip_sync:
        console.print("\n[bold cyan]📦 Syncing vector database before triage...[/bold cyan]")
        try:
            sync_recent_emails(max_emails=50)
            console.print("[bold green]✅ Vector sync complete.[/bold green]\n")
        except Exception as e:
            console.print(f"[bold yellow]⚠ Vector sync encountered an issue: {e}[/bold yellow]")
            console.print("[bold yellow]  Continuing with triage anyway...[/bold yellow]\n")
    else:
        console.print("[dim]⏭ Vector sync skipped (--skip-sync).[/dim]")

    # --- 3. Run triage ---
    console.print(f"📄 [bold]File:[/bold] {file}")
    console.print(f"🔍 [bold]Keywords:[/bold] {keyword_str}")
    console.print(f"📅 [bold]Timeframe:[/bold] {timeframe} | 🔢 [bold]Max Emails:[/bold] {max_results}")
    if label:
        console.print(f"🏷️  [bold]Label:[/bold] {label}")
    if tasks:
        console.print(f"📝 [bold]Tasks:[/bold] Enabled")
    if semantic:
        console.print(f"🧠 [bold]Search Mode:[/bold] Semantic (Vector DB)")

    state = {
        "messages": [HumanMessage(content="Automated Run Triggered")],
        "emails": [],
        "pending_send": None,
        "pending_delete": None,
        "pending_todo": None,
        "triage_config": {
            "keywords": keywords,
            "add_tasks": tasks,
            "label_to_add": label,
            "timeframe": timeframe,
            "max_results": max_results,
            "use_semantic": semantic,
        },
        "intent": "triage",
        "response": "",
    }

    with console.status("[cyan]🧠 Running workflow... (Fetching, reading, summarizing, and emailing)[/cyan]", spinner="dots"):
        final_event = None
        try:
            for event in graph.stream(state):
                final_event = event
        except Exception as e:
            console.print(f"\n[bold red]⚠ Error during triage:[/bold red] {e}")
            raise typer.Exit(1)

    if final_event:
        node_name = list(final_event.keys())[0]
        response_text = final_event[node_name].get("response", "Done.")

        console.print("\n[bold green]✅ Job Complete![/bold green]")
        console.print(Panel(response_text, title="[bold cyan]Execution Summary[/bold cyan]", border_style="cyan"))


if __name__ == "__main__":
    app()
