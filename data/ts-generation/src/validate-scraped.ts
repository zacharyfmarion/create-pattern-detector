#!/usr/bin/env bun
/**
 * Validate Scraped FOLD Files
 *
 * Validates FOLD files scraped from the web and organizes them:
 * - Valid files → ../output/scraped/validated/
 * - Invalid files → ../output/scraped/rejected/
 *
 * Uses existing validation logic from validators/
 *
 * Usage:
 *   bun run src/validate-scraped.ts
 *   bun run src/validate-scraped.ts --input ../output/scraped/raw
 */

import { readdir, readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { join, basename } from 'node:path';
import { parseArgs } from 'node:util';
import type { FOLDFormat } from './types/crease-pattern.ts';
import { validate } from './validators/validator.ts';

/**
 * Command line options
 */
interface CLIOptions {
  input: string;
  skipGlobal: boolean;
}

/**
 * Parse command line arguments
 */
function parseArguments(): CLIOptions {
  const { values } = parseArgs({
    options: {
      input: { type: 'string', short: 'i', default: '../output/scraped/raw' },
      'skip-global': { type: 'boolean', default: true }, // Default skip global for speed
    },
  });

  return {
    input: values.input as string,
    skipGlobal: values['skip-global'] || true,
  };
}

/**
 * Load and parse a FOLD file
 */
async function loadFOLDFile(filepath: string): Promise<{ fold: FOLDFormat | null; error: string | null }> {
  try {
    const content = await readFile(filepath, 'utf-8');
    const data = JSON.parse(content);

    // Basic structure check
    if (!data.vertices_coords || !Array.isArray(data.vertices_coords)) {
      return { fold: null, error: 'Missing or invalid vertices_coords' };
    }

    if (!data.edges_vertices || !Array.isArray(data.edges_vertices)) {
      return { fold: null, error: 'Missing or invalid edges_vertices' };
    }

    return { fold: data as FOLDFormat, error: null };
  } catch (e) {
    if (e instanceof SyntaxError) {
      return { fold: null, error: `Invalid JSON: ${e.message}` };
    }
    return { fold: null, error: `Failed to read file: ${e}` };
  }
}

/**
 * Validate scraped FOLD files
 */
async function main() {
  const options = parseArguments();

  const inputDir = options.input;
  const validatedDirS = '../output/scraped/validated/tier-s';
  const validatedDirA = '../output/scraped/validated/tier-a';
  const rejectedDir = '../output/scraped/rejected';

  console.log('🔍 Validating Scraped FOLD Files');
  console.log('=====================================');
  console.log(`Input: ${inputDir}`);
  console.log(`Skip global validation: ${options.skipGlobal}`);
  console.log('');

  // Ensure output directories exist
  await mkdir(validatedDirS, { recursive: true });
  await mkdir(validatedDirA, { recursive: true });
  await mkdir(rejectedDir, { recursive: true });

  // Find all .fold files
  let files: string[];
  try {
    const entries = await readdir(inputDir);
    files = entries.filter(f => f.endsWith('.fold'));
  } catch (e) {
    console.error(`❌ Failed to read input directory: ${inputDir}`);
    console.error(`   ${e}`);
    console.error('');
    console.error(`Please create ${inputDir}/ and add your scraped .fold files there.`);
    process.exit(1);
  }

  if (files.length === 0) {
    console.log(`No .fold files found in ${inputDir}/`);
    console.log('');
    console.log(`Please add scraped .fold files to ${inputDir}/ and run again.`);
    process.exit(0);
  }

  console.log(`Found ${files.length} FOLD files`);
  console.log('');

  let validCount = 0;
  let invalidCount = 0;

  for (let i = 0; i < files.length; i++) {
    const filename = files[i];
    const filepath = join(inputDir, filename);

    console.log(`[${i + 1}/${files.length}] Processing ${filename}...`);

    // Load file
    const { fold, error: loadError } = await loadFOLDFile(filepath);

    if (!fold || loadError) {
      // Structural error - reject immediately
      const destPath = join(rejectedDir, filename);
      await copyFile(filepath, destPath);

      const errorPath = join(rejectedDir, `${basename(filename, '.fold')}.errors.txt`);
      await writeFile(errorPath, `Validation errors for ${filename}\n${'='.repeat(60)}\n\nERROR: ${loadError}\n`);

      console.log(`❌ ${filename}`);
      console.log(`   ❌ ${loadError}`);
      console.log('');

      invalidCount++;
      continue;
    }

    // Validate using existing validators
    const result = await validate(fold, { skipGlobal: options.skipGlobal });

    const numVertices = fold.vertices_coords?.length || 0;
    const numEdges = fold.edges_vertices?.length || 0;
    const numCreases = fold.edges_assignment?.filter(a => ['M', 'V', 'F', 'U'].includes(a)).length || 0;

    if (result.tier === 'S' || result.tier === 'A') {
      // Valid - copy to appropriate tier directory
      const tierDir = result.tier === 'S' ? validatedDirS : validatedDirA;
      const destPath = join(tierDir, filename);
      await copyFile(filepath, destPath);

      // Save validation metadata
      const metadataPath = join(tierDir, `${basename(filename, '.fold')}.json`);
      await writeFile(metadataPath, JSON.stringify({
        filename,
        validation: result,
        stats: { numVertices, numEdges, numCreases },
        validated_at: new Date().toISOString(),
      }, null, 2));

      console.log(`✅ ${filename} (Tier ${result.tier})`);
      console.log(`   Vertices: ${numVertices}, Edges: ${numEdges}, Creases: ${numCreases}`);
      console.log(`   Checks passed: ${result.passed.join(', ')}`);
      console.log('');

      validCount++;
    } else {
      // Invalid - copy to rejected/
      const destPath = join(rejectedDir, filename);
      await copyFile(filepath, destPath);

      const errorPath = join(rejectedDir, `${basename(filename, '.fold')}.errors.txt`);
      const errorContent = [
        `Validation errors for ${filename}`,
        '='.repeat(60),
        '',
        ...result.errors.map(e => `ERROR: ${e}`),
        '',
        `Checks passed: ${result.passed.join(', ')}`,
        `Checks failed: ${result.failed.join(', ')}`,
      ].join('\n');
      await writeFile(errorPath, errorContent);

      console.log(`❌ ${filename}`);
      result.errors.slice(0, 3).forEach(err => {
        console.log(`   ❌ ${err}`);
      });
      if (result.errors.length > 3) {
        console.log(`   ... and ${result.errors.length - 3} more errors (see ${basename(errorPath)})`);
      }
      console.log('');

      invalidCount++;
    }
  }

  // Summary
  console.log('='.repeat(60));
  console.log('✅ Validation Complete!');
  console.log(`   Valid: ${validCount} → ../output/scraped/validated/`);
  console.log(`   Invalid: ${invalidCount} → ${rejectedDir}/`);

  if (validCount > 0) {
    console.log('');
    console.log('💡 Next step: Render validated files with:');
    console.log('   cd ../..');
    console.log('   python scripts/render_scraped.py');
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
