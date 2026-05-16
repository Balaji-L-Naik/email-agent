"""
main.py
-------
Entry point for the Email Agent CLI.
Run with:  python main.py
"""

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import print as rprint
from langchain_core.messages import HumanMessage

from agent.graph import graph

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help="Local Email Agent powered by Ollama + Gmail")
console = Console()

BANNER = Text.assemble(
    ("  ✉  ", "bold cyan"),
    ("Email Agent", "bold white"),
    ("  ✉  ", "bold cyan"),
    justify="center",
)

SUBTITLE = Text(
    "Local · Private · Agent",
    style="dim",
    justify="center",
)


def print_banner():
    console.print()
    console.print(Panel.fit(
        Text.assemble(BANNER, "\n", SUBTITLE),
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print(Rule(style="dim"))
    console.print(
        "  Type your request in plain English. "
        "[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to leave.[/dim]"
    )
    console.print(Rule(style="dim"))
    console.print()


def print_response(text: str):
    console.print()
    console.print(Panel(
        text,
        title="[bold cyan]Agent[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


# ---------------------------------------------------------------------------
# State initialiser
# ---------------------------------------------------------------------------

def initial_state():
    return {
        "messages": [],
        "emails": [],
        "pending_send": None,
        "intent": "converse",
        "response": "",
    }


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@app.command()
def main():
    """Start the interactive email agent."""
    print_banner()

    state = initial_state()

    while True:
        try:
            raw = console.input("[bold cyan]You ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not raw:
            continue

        if raw.lower() in {"exit", "quit", "bye", "q"}:
            console.print("\n[bold cyan]Goodbye! Have a great day. ✉[/bold cyan]\n")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=raw)]

        # Run through graph with dynamic execution updates
        with console.status("[cyan]🧠 Understanding request…[/cyan]", spinner="dots") as status:
            try:
                for event in graph.stream(state):
                    # event is a dict { node_name: state_updates }
                    node_name = list(event.keys())[0]
                    node_state = event[node_name]
                    
                    # Merge updates into local state
                    state.update(node_state)
                    
                    # Adapt the status message dynamically based on the current context
                    if node_name == "classify_intent":
                        intent = state.get("intent", "")
                        if state.get("pending_send") and state["pending_send"].get("awaiting_confirm"):
                            status.update("[blue]🚀 Dispatching email via Gmail API…[/blue]")
                        elif state.get("pending_delete") and state["pending_delete"].get("awaiting_confirm"):
                            status.update("[blue]🗑️ Moving selected emails to trash…[/blue]")
                        elif intent == "list_search":
                            status.update("[cyan]📨 Searching and fetching emails…[/cyan]")
                        elif intent == "read":
                            status.update("[cyan]📖 Fetching full email content…[/cyan]")
                        elif intent == "summarize":
                            status.update("[cyan]📝 Analyzing emails and generating summary…[/cyan]")
                        elif intent == "send":
                            status.update("[cyan]🚀 Extracting email fields and preparing draft…[/cyan]")
                        elif intent == "delete":
                            status.update("[cyan]🗑️ Filtering emails for deletion…[/cyan]")
                        elif intent == "label":
                            status.update("[cyan]🏷️ Applying labels to email…[/cyan]")
                        elif intent == "converse":
                            status.update("[cyan]💬 Generating response…[/cyan]")

            except Exception as exc:
                print_response(f"[red]⚠ An error occurred:[/red] {exc}\n\nPlease try again.")
                continue

        response = state.get("response", "")
        if response:
            print_response(response)
        else:
            print_response("[dim](No response generated — please try again.)[/dim]")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()