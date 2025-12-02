# Resume Dataset Generation

The dataset generation script is now **fully resumable**. If your process crashes or is interrupted, you can simply restart it with the same command and it will pick up where it left off.

## How It Works

The script automatically:
1. Scans the output directory for existing patterns
2. Extracts the index numbers from filenames (`cp-<timestamp>-<INDEX>.fold`)
3. Skips generating any indices that already exist
4. Continues from where it left off

## Example

If you're running:
```bash
bun run src/generate-dataset.ts --count 80000 --output ../output/synthetic/raw --min-creases 30 --max-creases 800
```

And it crashes after generating 5,000 patterns, simply run the **exact same command** again:
```bash
bun run src/generate-dataset.ts --count 80000 --output ../output/synthetic/raw --min-creases 30 --max-creases 800
```

The script will:
- Find the 5,000 existing patterns
- Skip indices 0-4,999
- Resume generating from index 5,000 onwards
- Show you: `📂 Found 5000 existing patterns - resuming generation`

## Important Notes

1. **Use the same parameters**: Make sure to use the exact same `--count`, `--output`, and other parameters
2. **No data loss**: Existing files are never overwritten or deleted
3. **Fast startup**: The script scans existing files quickly, even with thousands of patterns
4. **Works across restarts**: Survives computer restarts, crashes, or manual interruptions (Ctrl+C)
5. **Tier-aware**: Properly handles patterns in `tier-a`, `tier-s`, and `rejected` directories

## Progress Tracking

When resuming, you'll see output like:
```
📂 Found 5000 existing patterns - resuming generation

[5001/80000] Generating cp-1761535860349-005000...
  ✓ A tier (5 checks passed)
```

Final summary will show:
```
Total patterns: 80000
  - Generated this session: 75000
  - Skipped (already existed): 5000
```

## Safety

- The script never modifies or deletes existing files
- Each index is unique and generated only once
- You can safely Ctrl+C at any time and resume later
