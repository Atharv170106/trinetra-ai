# CLAUDE.md

Project: **Trinetra AI** — air-gapped satellite semantic search + multi-temporal change analysis. SIH 2026, PS 26227 (Ministry of Defence / DGIS).

## Communication Style & Formatting Rules

These are strict. They override any default verbosity.

- **Be concise.** No introductory fluff, no conversational filler, no restating the question back. Lead with the answer.
- **Default to bullet points** for all explanations, steps, and comparisons.
- **No long paragraphs.** Break everything into highly scannable lists. If a thought needs more than two sentences, it needs to be bullets instead.
- **When providing code:** output the code, then a bulleted list of what changed. Nothing else. No preamble explaining what you are about to write, no postamble summarising what was just written.
- **No basic-concept explanations.** Assume senior-level Python, React, Docker and geospatial knowledge.
- **Skip the postamble.** When a task is done, say it is done. Do not recap.

## Hard Constraints

- **Hardware:** RTX 4060 Laptop, 8 GB VRAM, 16 GB RAM. Never load full GeoTIFFs into memory — windowed reads via `rasterio` only.
- **Air-gap:** 100% offline via Docker. No external APIs, cloud DBs, or CDN map tiles in the deployed stack. Online-only tooling must be a separate staging script excluded from the image.
- **Phased execution:** One phase or module per request. Never emit the whole codebase.
- **No hallucinated APIs:** If library syntax is uncertain, emit `[RESEARCH_REQUEST]: "..."` for the user to run against a web-connected model instead of guessing.
