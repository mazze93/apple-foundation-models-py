---
name: afm
description: Offload bounded text transforms to Apple's on-device model via the `afm` CLI — free, offline, and the input never enters Claude's context. Use this whenever a task means running the same shallow transform over many items (classify, tag, extract fields, summarize, triage, draft short text across dozens of files, notes, logs, emails, or rows), or when the user wants material processed without Claude reading it ("keep this local", "don't send this to the cloud", "private", "sensitive"), or to save context on large inputs where only a digest is needed. Also use it when the user mentions afm, the on-device model, or Apple Intelligence for a task. Not for writing or reviewing code, multi-step reasoning, or anything that needs tools — keep those on Claude, and send agentic offline work to claude-local instead.
---

# afm — the on-device text lane

`afm` runs Apple's on-device Foundation Model (~3B parameters, Apple Intelligence) as a shell command: text in, text out, one call, no tools. It costs nothing, works offline, and — the part that matters most — whatever you pipe into it with `<` is processed on the Mac without ever passing through your context.

It is a *small* model. Treat it like a fast, literal intern: excellent at "label this", "pull these fields out", "summarize this in three bullets"; unreliable at anything that needs judgment, code, or arithmetic over many steps. The job of this skill is to route the right work to it and keep its output trustworthy.

## Decide: afm, claude-local, or you

| Work | Lane |
|---|---|
| Same shallow transform over many items; private text you shouldn't read; digest of a big input | **afm** |
| Multi-step offline work that needs tools or file edits (agent loop) | **claude-local** (`claude-local -p "..."`) |
| Code, review, reasoning, final user-facing writing, anything subtle | **you** |

If the per-item decision is something you'd want to double-check yourself, it doesn't belong in afm. Cheap is only worth it when the answers are right.

## Before the first call

```bash
afm status    # exit 0 = ready; prints context window size and token-counting support
```

- `command not found` → install it: `uv tool install --editable ~/Projects/tools/apple-foundation-models-py`
- exit 1 → Apple Intelligence is off or still downloading; `status` prints why. Tell the user; don't work around it.

## How to call it

```bash
afm --no-stream -t 0 -m 200 "INSTRUCTION" < input.txt        # plain text result
afm --schema schema.json -t 0 "INSTRUCTION" < input.txt      # JSON result, guaranteed shape
afm -s "SYSTEM INSTRUCTIONS" --no-stream "INSTRUCTION" < input.txt
```

- **Pipe inputs with `<`, don't `cat` them into your context first.** Reading the file yourself defeats the privacy and context savings.
- **Use `--schema` whenever you'll parse the result.** Free-text replies come with chatty preambles ("I am a foundation model created by Apple…"); a schema forces exactly the fields you asked for. Constrain labels with `enum` so the model can't invent categories:
  ```json
  {"type": "object",
   "properties": {"label": {"type": "string", "enum": ["bug", "feature", "question"]},
                  "summary": {"type": "string"}},
   "required": ["label", "summary"]}
  ```
- `-t 0` for repeatable output; `--no-stream` in scripts; `-m` sizes the response (and is reserved out of the window, so keep it no bigger than you need).
- Piped text is appended after the instruction words, so put the instruction on the command line and the material on stdin.

Speed on an M-series Mac: roughly 1–2 s for a short classification, ~4 s to summarize ~2k tokens. Run calls sequentially in a loop; write results to a file and read the file, so a long batch doesn't flood your context.

```bash
for f in notes/*.md; do
  printf '%s\t' "$f"
  afm --schema label.json -t 0 -m 60 "Classify this note." < "$f" | tr -d '\n'
  echo
done > /tmp/labels.tsv
```

## Limits and what each failure means

| Exit | Meaning | Do this |
|---|---|---|
| 0 | ok | spot-check (below) |
| 1 | unavailable, or a generation error (message names it) | see `GuardrailViolationError` note; otherwise report |
| 2 | usage error, or **input too large** — afm counted tokens and refused before sending | chunk it (below); `--truncate` only if the head alone is enough |
| 130 | interrupted | — |

- **Context window is 8192 tokens on macOS 27** (`afm status` shows the real number) — shared by instructions, input, and response. About 7k tokens of input fit at the default `-m 1024`; roughly 4 characters per token for English.
- **Too big → map-reduce, not `--truncate`.** `--truncate` keeps only the beginning, which silently drops the rest of a document. To digest the whole thing, split it (e.g. `split -l 150 big.txt part_`), summarize each part, then summarize the summaries.
- **`GuardrailViolationError` can be a false positive.** Apple's safety filter has refused plainly benign, repetitive text. Don't retry in a loop or rephrase to sneak past it — record the item as unprocessed and tell the user which ones failed.

## Keep the output honest

A 3B model is confidently wrong more often than you are. Before you build on a batch:

- Spot-check a sample against the source — open 2–3 of the inputs yourself (when they aren't the private ones) and compare.
- For classification, look at the label distribution; one label swallowing everything usually means the instruction or enum is off.
- Present afm output to the user as what it is ("the on-device model labelled these"), not as your own verified conclusion.

## Privacy, stated precisely

The input stays on the Mac **only if you never read it** — pipe it with `<` and work from afm's output. afm's *output* does enter your context (and so the cloud) once you read it. When the user wants something kept local, say exactly that: "the files were processed on-device; I only saw the labels." If even the output is too sensitive for you to see, have afm write to a file the user opens themselves, and don't read it.
