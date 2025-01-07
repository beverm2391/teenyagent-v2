## Command Execution Rules
### Auto-Execution Criteria
Commands must be:
- Simple and predictable in outcome
- Non-destructive
- Limited to project scope
- Idempotent when possible

### Allowlist
- `pip install/uninstall` (specific package)
- `pytest` (with specific paths, no -v flag unless requested)
- `git status/log/diff` (read-only operations)
- `ls/pwd/cat` (read-only operations)
- `mkdir -p` (safe directory creation)
- `commit "type(scope): message" "- detail 1" "- detail 2"` (custom git add/commit/push)

### Denylist
- Any `rm/del` commands
- Any `sudo` operations
- Global pip operations without `-t`
- Force flags (`-f`, `--force`)
- Shell operations (`>&`, `|`, `;`)
- Environment modifications
- Network operations (`curl`, `wget`)


## Package Management
- Install packages directly in environment with pip
- Do not maintain requirements.txt - we work in the environment directly
- Only document critical version dependencies if absolutely necessary

## Code Style
- Keep it simple
- Less code is better
- Challenge complexity 

- using pytest
- when you need a new package, run its intsll command in the rood dir. do not every write a `requirements.txt` file or a dep in a `package.json` file.
- when you need to update a package, run its update command in the root dir.