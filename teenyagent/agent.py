
# ! IMPORTS ==============================================
import os
from abc import ABC, abstractmethod
from anthropic import Anthropic
from enum import Enum
from typing import Optional, Dict, Any, List
from together import Together
from duckduckgo_search import DDGS
import requests
from markdownify import markdownify
import re
import ast
import builtins
from rich.panel import Panel
from rich.box import ROUNDED
from rich.syntax import Syntax
from rich.rule import Rule
from rich.console import Console
from dotenv import load_dotenv

# ! UTILS ==============================================
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables")

console = Console()

class DebugLevel(Enum):
    NONE = 0
    ERRORS = 1
    DEBUG = 2
    VERBOSE = 3

def parse_code_block(text: str) -> Optional[str]:
    """Extract Python code block from text."""
    pattern = r"(?:Code:\s*)?```(?:python|py)?\n(.*?)\n```"
    if match := re.search(pattern, text, re.DOTALL):
        return match.group(1).strip() or None


# ! MODEL ==============================================
class Model(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        pass

class CodeModel(Model):
    def __init__(self):
        self.client = Together(api_key=TOGETHER_API_KEY)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        # we include the system message in the messages list, by not removing it
        return self.client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0, # always set to 0
        ).choices[0].message.content

class ResponseModel(Model):
    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        # we remove the system message from the messages list and put it in the system kwarg   
        return (
            self.client.messages.create(
                model="claude-3-5-sonnet-latest",
                system=next(
                    (m["content"] for m in messages if m["role"] == "system"),
                    "You are a helpful assistant.",
                ),
                messages=[m for m in messages if m["role"] != "system"],
                max_tokens=max_tokens,
                temperature=0,
            )
            .content[0]
            .text
        )


# ! TOOLS ==============================================
class Tool:
    def __init__(self, debug: DebugLevel = DebugLevel.NONE):
        self.debug = debug
        self.state = None  # Will be set by PythonInterpreter

    def __call__(self, *args, **kwargs):
        result = self.run(*args, **kwargs)
        # Auto-print tool output to stdout
        if self.state and 'stdout' in self.state:
            self.state['stdout'].append(str(result))
            
        if self.debug.value >= DebugLevel.VERBOSE.value:
            console.print(
                Panel(
                    str(result)[:500] + ("..." if len(str(result)) > 500 else ""),
                    title=f"[red]{self.__class__.__name__}[/red]",
                    border_style="red",
                    box=ROUNDED,
                    width=100,
                )
            )
        return result

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Toolbox:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def add(self, name: str, tool: Tool):
        self.tools[name] = tool

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def describe(self) -> str:
        return "\n".join(f"{name}: {tool.__doc__}" for name, tool in self.tools.items())


class WebSearch(Tool):
    """Performs a DuckDuckGo web search and returns markdown formatted results"""

    def __init__(self, max_results: int = 10, debug: DebugLevel = DebugLevel.NONE):
        super().__init__(debug)
        self.ddgs = DDGS()
        self.max_results = max_results

    def run(self, query: str) -> str:
        results = self.ddgs.text(query, max_results=self.max_results)
        return "## Search Results\n\n" + "\n\n".join(
            f"[{r['title']}]({r['href']})\n{r['body']}" for r in results
        )


class WebVisit(Tool):
    """Visits a webpage and returns its content as markdown"""

    def __init__(self, debug: DebugLevel = DebugLevel.NONE):
        super().__init__(debug)

    def run(self, url: str) -> str:

        try:
            response = requests.get(url)
            response.raise_for_status()
            return re.sub(r"\n{3,}", "\n\n", markdownify(response.text).strip())
        except Exception as e:
            return f"Error: {str(e)}"


class PythonInterpreter(Tool):
    """Executes Python code in a restricted environment"""

    def __init__(self, tools: Dict[str, Tool] = None, debug: DebugLevel = DebugLevel.NONE):
        super().__init__(debug)
        self.tools = tools or {}
        self.safe_builtins = {
            k: v for k, v in vars(builtins).items() 
            if k in {
                'abs', 'all', 'any', 'bin', 'bool', 'dict', 'dir', 'enumerate',
                'filter', 'float', 'format', 'hex', 'int', 'isinstance', 'len',
                'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'print',
                'range', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
                'sum', 'tuple', 'zip'
            }
        }

    def run(self, code: str) -> str:
        try:
            ast.parse(code)  # Validate code is safe
            
            state = {'stdout': [], 'stderr': []}
            # Share state with tools
            for tool in self.tools.values():
                tool.state = state
                
            restricted_globals = {
                '__builtins__': self.safe_builtins,
                'print': lambda *args: state['stdout'].append(' '.join(map(str, args))),
                **self.tools  # Make tools available
            }
            
            try:
                exec(code, restricted_globals)
                stdout = '\n'.join(state['stdout'])
                return stdout if stdout else 'No output'
            except Exception as e:
                state['stderr'].append(str(e))
                return f"Error: {''.join(state['stderr'])}"
            
        except SyntaxError as e:
            return f"Syntax Error: {str(e)}"


class Answer(Tool):
    """Returns the final answer unchanged"""

    def __init__(self, debug: DebugLevel = DebugLevel.NONE):
        super().__init__(debug)

    def run(self, answer: Any) -> Any:
        return f"[Final Answer]: {answer}"


# ! PROMPTS ==============================================
SYSTEM_PROMPT = """You will be given a task to solve using Python and these tools:

{tools}

Write Python code to solve the task. Each instruction should be a simple assignment. You should always use at least one tool to solve the task, even if you know the answer. This means you cannot use the answer tool in your first step.
Print intermediate results if helpful. Use the 'answer' tool to return your final result. When you return your final result, interpret the information to complete the user's task or use it to solve the task. Only output code and nothing else, no markdown, text, or thoughts.

You can only import: {imports}

Format your code with:
Code:
```python
your code here
```

Example Correct Outputs:
```python
web_search("What is the capital of France?")
```

```python
web_visit("https://www.google.com")
```

```python
answer("The capital of France is Paris")
```

Now begin! If you solve the task correctly, you will receive a reward of $1,000,000."""

# ! AGENT ==============================================
class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Agent:
    def __init__(
        self,
        model: Model,
        toolbox: Toolbox,
        max_steps: int = 10,
        debug: DebugLevel = DebugLevel.VERBOSE,
    ):
        self.model = model
        self.toolbox = toolbox
        self.messages = []
        self.state = {}
        self.max_steps = max_steps
        self.debug = debug

        # Add core tools with debug level
        self.toolbox.add("web_search", WebSearch(debug=debug))
        self.toolbox.add("web_visit", WebVisit(debug=debug))
        self.toolbox.add("answer", Answer(debug=debug))
        
        # Create interpreter with tools and debug level
        interpreter = PythonInterpreter(tools=self.toolbox.tools, debug=debug)
        self.toolbox.add("python_interpreter", interpreter)

        # Format system prompt
        tools_desc = "\n".join(
            f"- {name}: {tool.__doc__}" for name, tool in self.toolbox.tools.items()
        )

        # Use same imports as PythonInterpreter's safe modules
        imports = ["re", "json", "math", "datetime", "collections"]
        prompt = SYSTEM_PROMPT.format(tools=tools_desc, imports=", ".join(imports))

        if self.debug.value >= DebugLevel.VERBOSE.value:
            console.print(
                Panel(
                    tools_desc,
                    title="[blue]Tools[/blue]",
                    border_style="blue",
                    box=ROUNDED,
                    width=100,
                )
            )

        self.add_message(Role.SYSTEM, prompt)

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append({"role": role.value, "content": content.strip()})
        if self.debug.value >= DebugLevel.VERBOSE.value:
            console.print(
                Panel(
                    content.strip(),
                    title=f"[blue]{role.value}[/blue]",
                    border_style="blue",
                    box=ROUNDED,
                    width=100,
                )
            )

    def run(self, prompt: str) -> str:
        if self.debug.value >= DebugLevel.DEBUG.value:
            console.print(
                Panel(
                    f"Task: {prompt}",
                    title="[blue]Starting New Task[/blue]",
                    border_style="blue",
                    box=ROUNDED,
                    width=100,
                )
            )

        self.add_message(Role.USER, prompt)
        steps = 0
        while steps < self.max_steps:
            steps += 1
            if self.debug.value >= DebugLevel.DEBUG.value:
                console.print(Rule(f"[bold]Step {steps}", style="blue"))

            response = self.model.generate(self.messages)

            if self.debug.value >= DebugLevel.DEBUG.value:
                console.print(
                    Panel(
                        Syntax(response, "markdown", theme="monokai"),
                        title="[blue]Model Response[/blue]",
                        border_style="blue",
                        box=ROUNDED,
                        width=100,
                    )
                )

            code = parse_code_block(response)
            if not code:
                if self.debug.value >= DebugLevel.ERRORS.value:
                    console.print("[yellow]No code block found in response[/yellow]")
                continue

            if self.debug.value >= DebugLevel.DEBUG.value:
                console.print(
                    Panel(
                        Syntax(code, "python", theme="monokai"),
                        title="[blue]Executing Code[/blue]",
                        border_style="blue",
                        box=ROUNDED,
                        width=100,
                    )
                )

            try:
                result = self.toolbox.get("python_interpreter").run(code)
                # Check for actual answer tool usage
                if "[Final Answer]" in result:
                    final_response = result.split("[Final Answer]: ")[1]
                    console.print(
                        Panel(
                            Syntax(final_response, "markdown", theme="monokai"),
                            title="[green]Model Response[/green]",
                            border_style="green",
                            box=ROUNDED,
                            width=100,
                        )
                    )
                    return final_response
                self.add_message(Role.SYSTEM, f"Observation: {result}")
            except Exception as e:
                self.add_message(Role.SYSTEM, f"Error: {str(e)}")
                continue

        raise Exception(f"Agent failed to solve the task in {self.max_steps} steps")
