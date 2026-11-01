# `ezpz completion`

Emit shell completion for `ezpz` — `bash`, `zsh`, or `fish`.

```bash
ezpz completion bash   # print bash completion script
ezpz completion zsh    # print zsh completion script
ezpz completion fish   # print fish completion script
```

Redirect to your shell's completion directory, or use the `--install` flag where supported. Uses Click's built-in `click.shell_completion`; no `shtab` needed.

## Options

| Option | Description |
|--------|-------------|
| `SHELL` | One of `bash`, `zsh`, `fish`. |
| `--install` | Install to the shell's completion path (where supported). |
| `--help` | Show completion options. |

Requires `click>=8` (already pinned by `ezpz`).
