# Clean BP Reference Manifest

This directory is reserved for calibration-only clean digital box-pleat crease-pattern references.

Reference rows are not training labels. They are used by `bun run bp-realism-report` to compare generated samples against real/BPStudio-like visual distributions.

Recommended JSONL fields:

```json
{"id":"designer-or-source-id","imagePath":"images/example.png","sourceUrl":"https://example.com","license":"calibration-only","archetype":"insect","notes":"clean digital CP reference"}
```

Do not add copyrighted CP images here unless the source and allowed use are clear.
