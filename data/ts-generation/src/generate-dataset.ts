#!/usr/bin/env bun
/**
 * Main Dataset Generation Script
 *
 * Generates synthetic crease pattern dataset with validation.
 *
 * Usage:
 *   bun run src/generate-dataset.ts --count 100 --output ./test-dataset
 */

import { parseArgs } from 'node:util';
import { mkdir, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { existsSync } from 'node:fs';
import type { GenerationConfig, DatasetEntry, SymmetryType } from './types/crease-pattern.ts';
import { generateRabbitEarCP } from './generators/rabbit-ear-correct.ts';
import { applySymmetry } from './generators/symmetric.ts';
import { validate } from './validators/validator.ts';
import { saveFOLD } from './utils/fold-helpers.ts';

/**
 * Command line options
 */
interface CLIOptions {
  count: number;
  output: string;
  method: 'mixed' | 'tree' | 'box-pleating' | 'classic-bases';
  symmetry: 'mixed' | 'none' | '2-fold' | '4-fold';
  width: number;
  height: number;
  minCreases: number;
  maxCreases: number;
  skipGlobal: boolean;
  parallel: boolean;
}

/**
 * Parse command line arguments
 */
function parseArguments(): CLIOptions {
  const { values } = parseArgs({
    options: {
      count: { type: 'string', short: 'c', default: '100' },
      output: { type: 'string', short: 'o', default: '../output/synthetic/raw' },
      method: { type: 'string', default: 'mixed' },
      symmetry: { type: 'string', default: 'mixed' },
      width: { type: 'string', default: '1024' },
      height: { type: 'string', default: '1024' },
      'min-creases': { type: 'string', default: '10' },
      'max-creases': { type: 'string', default: '50' },
      'skip-global': { type: 'boolean', default: false },
      parallel: { type: 'boolean', default: false },
    },
  });

  return {
    count: parseInt(values.count as string),
    output: values.output as string,
    method: (values.method as any) || 'mixed',
    symmetry: (values.symmetry as any) || 'mixed',
    width: parseInt(values.width as string),
    height: parseInt(values.height as string),
    minCreases: parseInt(values['min-creases'] as string),
    maxCreases: parseInt(values['max-creases'] as string),
    skipGlobal: values['skip-global'] || false,
    parallel: values.parallel || false,
  };
}

/**
 * Generate a single crease pattern
 */
function generateSingleCP(
  index: number,
  options: CLIOptions
): { config: GenerationConfig; fold: any } {
  // Use Rabbit Ear generator (combines axioms + single-vertex methods)
  const method = 'rabbit-ear';

  // Determine symmetry
  let symmetry: SymmetryType;

  if (options.symmetry === 'mixed') {
    const r = Math.random();
    if (r < 0.4) symmetry = '2-fold';
    else if (r < 0.4) symmetry = '4-fold';
    else symmetry = 'none';
  } else {
    symmetry = options.symmetry as SymmetryType;
  }

  // Determine crease count
  const numCreases = Math.floor(
    Math.random() * (options.maxCreases - options.minCreases) + options.minCreases
  );

  // Create config
  const config: GenerationConfig = {
    numCreases,
    width: options.width,
    height: options.height,
    symmetry,
    seed: Date.now() + index,
    method,
  };

  // Generate pattern using Rabbit Ear
  let fold = generateRabbitEarCP(config);

  // Apply symmetry
  fold = applySymmetry(fold, symmetry);

  return { config, fold };
}

/**
 * Get existing pattern indices to support resume
 */
async function getExistingIndices(outputDir: string): Promise<Set<number>> {
  const indices = new Set<number>();

  const dirs = ['tier-s', 'tier-a', 'rejected'];

  for (const dir of dirs) {
    const fullPath = join(outputDir, dir);
    if (!existsSync(fullPath)) continue;

    try {
      const files = await readdir(fullPath);
      for (const file of files) {
        // Extract index from filename: cp-<timestamp>-<index>.fold
        if (file.endsWith('.fold')) {
          const match = file.match(/cp-\d+-(\d+)\.fold$/);
          if (match) {
            indices.add(parseInt(match[1]));
          }
        }
      }
    } catch (err) {
      // Directory doesn't exist yet, that's fine
    }
  }

  return indices;
}

/**
 * Main generation loop
 */
async function main() {
  const options = parseArguments();

  console.log('🎨 Synthetic Crease Pattern Generator');
  console.log('=====================================');
  console.log(`Target count: ${options.count}`);
  console.log(`Output directory: ${options.output}`);
  console.log(`Method: ${options.method}`);
  console.log(`Symmetry: ${options.symmetry}`);
  console.log(`Size: ${options.width}x${options.height}`);
  console.log(`Crease range: ${options.minCreases}-${options.maxCreases}`);
  console.log(`Skip global validation: ${options.skipGlobal}`);
  console.log('');

  // Create output directory
  await mkdir(options.output, { recursive: true });
  await mkdir(join(options.output, 'fold'), { recursive: true });
  await mkdir(join(options.output, 'tier-s'), { recursive: true });
  await mkdir(join(options.output, 'tier-a'), { recursive: true });
  await mkdir(join(options.output, 'rejected'), { recursive: true });

  // Check for existing patterns (resume support)
  const existingIndices = await getExistingIndices(options.output);
  const alreadyGenerated = existingIndices.size;

  if (alreadyGenerated > 0) {
    console.log(`📂 Found ${alreadyGenerated} existing patterns - resuming generation`);
    console.log('');
  }

  const stats = {
    generated: 0,
    tierS: 0,
    tierA: 0,
    rejected: 0,
    skipped: alreadyGenerated,
  };

  const startTime = Date.now();

  // Generation loop
  for (let i = 0; i < options.count; i++) {
    // Check if this index already exists (resume support)
    if (existingIndices.has(i)) {
      if (i < 10 || i % 100 === 0) {
        console.log(`[${i + 1}/${options.count}] Skipping index ${i} (already exists)`);
      }
      continue;
    }

    const id = `cp-${Date.now()}-${i.toString().padStart(6, '0')}`;

    console.log(`[${i + 1}/${options.count}] Generating ${id}...`);

    try {
      // Generate
      const { config, fold } = generateSingleCP(i, options);

      // Validate
      const validation = await validate(fold, {
        skipGlobal: options.skipGlobal,
        skipFlatFolder: true, // Flat-folder is optional
      });

      // Create dataset entry
      const entry: DatasetEntry = {
        id,
        fold,
        validation,
        config,
        timestamp: new Date().toISOString(),
      };

      // Save to appropriate directory
      let tierDir: string;

      if (validation.tier === 'S') {
        tierDir = join(options.output, 'tier-s');
        stats.tierS++;
      } else if (validation.tier === 'A') {
        tierDir = join(options.output, 'tier-a');
        stats.tierA++;
      } else {
        tierDir = join(options.output, 'rejected');
        stats.rejected++;
      }

      // Save FOLD file
      const foldPath = join(tierDir, `${id}.fold`);
      await saveFOLD(fold, foldPath);

      // Save metadata
      const metaPath = join(tierDir, `${id}.json`);
      await Bun.write(metaPath, JSON.stringify(entry, null, 2));

      console.log(`  ✓ ${validation.tier} tier (${validation.passed.length} checks passed)`);

      stats.generated++;

    } catch (error) {
      console.error(`  ✗ Error: ${error}`);
      stats.rejected++;
    }

    // Progress update every 10
    if ((i + 1) % 10 === 0) {
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = (i + 1) / elapsed;
      const remaining = (options.count - i - 1) / rate;

      console.log('');
      console.log(`Progress: ${i + 1}/${options.count} (${((i + 1) / options.count * 100).toFixed(1)}%)`);
      console.log(`Tier S: ${stats.tierS} | Tier A: ${stats.tierA} | Rejected: ${stats.rejected}`);
      console.log(`Rate: ${rate.toFixed(2)}/sec | ETA: ${remaining.toFixed(0)}s`);
      console.log('');
    }
  }

  // Final statistics
  const totalTime = (Date.now() - startTime) / 1000;
  const totalPatterns = stats.generated + stats.skipped;

  console.log('');
  console.log('✅ Generation Complete!');
  console.log('======================');
  console.log(`Total patterns: ${totalPatterns}`);
  console.log(`  - Generated this session: ${stats.generated}`);
  console.log(`  - Skipped (already existed): ${stats.skipped}`);
  console.log('');
  console.log(`Tier S (gold): ${stats.tierS} (${stats.generated > 0 ? (stats.tierS / stats.generated * 100).toFixed(1) : 0}%)`);
  console.log(`Tier A (silver): ${stats.tierA} (${stats.generated > 0 ? (stats.tierA / stats.generated * 100).toFixed(1) : 0}%)`);
  console.log(`Rejected: ${stats.rejected} (${stats.generated > 0 ? (stats.rejected / stats.generated * 100).toFixed(1) : 0}%)`);
  console.log(`Total time: ${totalTime.toFixed(1)}s`);
  console.log(`Average rate: ${stats.generated > 0 ? (stats.generated / totalTime).toFixed(2) : 0}/sec`);
  console.log('');
  console.log(`Output saved to: ${options.output}`);
}

// Run main
main().catch(console.error);
