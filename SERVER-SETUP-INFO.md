# Strata server settings: sampling and the model

The settings below live in the config, `strata-unsloth-ud-q4_k_xl.json` in the Strata folder, not in server flags.
Restart the server (`run-unsloth-ud-q4_k_xl.bat`) after changing the config.

## Temperature, top-k and other sampling settings

Add a `"sampling"` block as a top-level key in the config, next to `"port"`:

```json
 "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
```

**Supported keys:** `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`,
`frequency_penalty`, `penalty_last_n` and `seed`.

**Limits:**

| Key | Allowed values |
| --- | --- |
| `temperature` | 0 or more (0 = always pick the most likely token) |
| `top_k` | a whole number from 1 to 64 |
| `top_p` | above 0, up to 1 |
| `min_p` | 0 to 1 |

The server refuses to start if a value is out of range. Keys it doesn't know are named at startup and ignored.

**Which setting wins**, highest first:

1. Values the client sends in its request (for example, if pi sends a temperature, pi's is used).
2. The settings on the server's web Chat page.
3. The config's `"sampling"` block.

**Re-running setup:** setup writes the config from scratch. `tools\v100_setup.py` (which the `v100\*.bat` steps
run) puts your `sampling`, `aliases` and `max_tokens` back afterwards. If you run `setup.py` directly, check the
block is still there.

## Pointing the server at a different model

Changing the model is a config change, not a flag. These entries in the config's `"args"` list are tied to the
current model:

| Entry | What it points to |
| --- | --- |
| `--native` | the first GGUF shard (`...-00001-of-0000N.gguf`) |
| `--pack` | the converted model folder setup builds, including the tokenizer |
| `--mtp` | the draft head used for speculative decoding |
| `--ple-gguf` | the separate PLE-table file `tools\v100_ple_split.py` made from the current model |
| `--expert-profile` | the expert ranking |

**A fine-tune that is still a GGUF of this same model** (same architecture):

1. Put all of its shards in one folder (`<name>-00001-of-0000N.gguf` ... `-0000N-of-0000N.gguf`).
2. Run setup with `--gguf-dir` pointing at that folder, so it builds the config and the converted model folder
   from it. Put the fine-tune's GGUF folder somewhere setup can find it, with only one model's shards in it.
   Setup also takes the quant type (`--model`, currently `UD-Q4_K_XL`). A fine-tune quantized differently may
   need checking first.
3. Run `v100\13_ple_split.bat` to make the fine-tune's own PLE-table file. The old one belongs to the current
   model.
4. The old draft head and expert ranking still work, though fewer drafts may be accepted.

A different architecture or file format needs more work than this.
