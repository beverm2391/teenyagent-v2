#!/usr/bin/env python3
"""Debug runner for end-to-end tests."""

import sys
import pytest
from rich.console import Console
from teenyagent.utils import debug, DebugLevel

console = Console()

def main():
    """Run a single test with maximum debugging."""
    if len(sys.argv) != 2:
        console.print("[red]Usage: python3 debug_test.py <test_name>[/red]")
        console.print("\nAvailable tests:")
        console.print("- test_simple_calculation")
        console.print("- test_web_search_and_answer")
        console.print("- test_multi_step_task")
        console.print("- test_error_handling")
        console.print("- test_state_persistence")
        sys.exit(1)
    
    test_name = sys.argv[1]
    
    with debug(DebugLevel.DEBUG):
        console.print(f"\n[cyan]Running test: {test_name}[/cyan]")
        console.print("[dim]Debug level: DEBUG[/dim]")
        console.print("[dim]=" * 80)
        
        # Run specific test with pytest
        pytest.main([
            "-v",
            "--no-header",
            "--capture=no",
            f"tests/integration/test_end_to_end.py::{test_name}"
        ])

if __name__ == "__main__":
    main() 