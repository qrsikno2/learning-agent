from rich.console import Console
from rich.panel import Panel

console = Console()


def print_user_input(text: str):
    console.print(Panel(text, title="User", style="bold cyan", border_style="cyan"))


def print_assistant_reply(text: str):
    console.print(Panel(text, title="Assistant", style="bold green", border_style="green"))


def print_tool_call(name: str, args):
    console.print(f"  [bold yellow]⚡ Tool Call:[/] [cyan]{name}[/]({args})")


def print_tool_result(name: str, result: str):
    preview = result[:300] + "..." if len(result) > 300 else result
    console.print(f"  [dim]📋 {name} result:[/] {preview}")


def print_final_answer(text: str):
    console.print(Panel(text, title="Final Answer", style="bold magenta", border_style="magenta"))
