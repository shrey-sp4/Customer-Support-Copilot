"""CLI Demo logic for the Copilot."""
import json
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def print_tool_trace(trace: list):
    """Pretty print the tool execution trace."""
    if not trace:
        console.print("[dim]No tool calls made.[/dim]")
        return
        
    for i, call in enumerate(trace, 1):
        tool = call.get("tool", "Unknown")
        args = call.get("args", {})
        res  = call.get("result", {})
        
        args_str = json.dumps(args, indent=2)
        res_str  = json.dumps(res, indent=2)
        
        # Truncate long results for display
        if len(res_str) > 500:
            res_str = res_str[:500] + "\n... [truncated]"
            
        panel = Panel(
            f"[bold cyan]Args:[/bold cyan]\n{args_str}\n\n[bold green]Result:[/bold green]\n{res_str}",
            title=f"Step {i}: {tool}",
            border_style="blue",
        )
        console.print(panel)


def interactive_loop(executor, show_trace: bool = True):
    """Run an interactive CLI chat loop."""
    console.print(Panel.fit("[bold green]Support Copilot CLI[/bold green]\nType 'quit' or 'exit' to stop.", border_style="green"))
    
    history = ""
    while True:
        try:
            query = console.input("\n[bold yellow]User:[/bold yellow] ")
            if query.lower() in ["quit", "exit"]:
                break
            if not query.strip():
                continue
                
            console.print("\n[dim]Thinking...[/dim]")
            result = executor.run(query, history=history)
            
            if show_trace:
                console.print("\n[bold magenta]--- Tool Trace ---[/bold magenta]")
                print_tool_trace(result.get("tool_trace", []))
            
            console.print("\n[bold cyan]--- Final Answer ---[/bold cyan]")
            console.print(result.get("final_answer", ""))
            console.print(f"\n[dim]Latency: {result.get('latency_ms', 0):.0f} ms | Decision: {result.get('decision')}[/dim]")
            
            # Append to simple history
            history += f" USER: {query} AGENT: {result.get('final_answer','')}"
            history = history[-500:] # Keep last 500 chars
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
