# Teenyagent v2
MANIFEST: this should be missing lots of features. the goal is to get a simple agent working and understand all the code.

- Tools
  - default tools only
    - duckduckgo search
    - visit webpage
    - local python code execution
    - final answer
- Code Agent
  - multi-step loop
    - loop until final answer or max steps
    - generate code to call tools
      - run code in local python interpreter
      - pass answer back
  - basic state dict (memory) for agent
  - can call tools with code (figure out how smolagents does this)??
- Rich console output with color
