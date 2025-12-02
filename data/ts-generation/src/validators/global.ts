/**
 * Global validation using FOLD CLI
 *
 * Uses the reference FOLD implementation to validate flat-foldability
 */

import { $ } from 'bun';
import type { FOLDFormat } from '../types/crease-pattern.ts';
import { saveFOLD } from '../utils/fold-helpers.ts';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/**
 * Validate using FOLD CLI --flat-fold probe
 *
 * @param fold FOLD format crease pattern
 * @returns True if passes global validation
 */
export async function validateGlobal(fold: FOLDFormat): Promise<{ valid: boolean; errors: string[] }> {
  const errors: string[] = [];

  try {
    // Save to temporary file
    const tmpFile = join(tmpdir(), `cp-validate-${Date.now()}.fold`);
    await saveFOLD(fold, tmpFile);

    // Run FOLD CLI with --flat-fold probe
    const result = await $`fold --flat-fold ${tmpFile}`.quiet();

    // Check output
    const output = result.stdout.toString();
    const exitCode = result.exitCode;

    if (exitCode !== 0) {
      errors.push('FOLD CLI validation failed (non-zero exit code)');
    }

    // Parse output for flat-fold result
    if (output.includes('false') || output.includes('not flat-foldable')) {
      errors.push('Pattern is not globally flat-foldable');
    }

    // Clean up temp file
    await $`rm -f ${tmpFile}`.quiet();

  } catch (error) {
    // If FOLD CLI is not installed or fails, record error
    errors.push(`Global validation error: ${error}`);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate using Flat-Folder (if available)
 *
 * @param fold FOLD format crease pattern
 * @returns True if passes flat-folder validation
 */
export async function validateFlatFolder(fold: FOLDFormat): Promise<{ valid: boolean; errors: string[] }> {
  const errors: string[] = [];

  try {
    // Check if flat-folder is installed
    const checkResult = await $`which flat-folder`.quiet();

    if (checkResult.exitCode !== 0) {
      // Flat-folder not installed, skip this check
      return { valid: true, errors: ['Flat-Folder not installed, skipping'] };
    }

    // Save to temporary file
    const tmpFile = join(tmpdir(), `cp-flatfolder-${Date.now()}.fold`);
    await saveFOLD(fold, tmpFile);

    // Run flat-folder
    const result = await $`flat-folder ${tmpFile}`.quiet();

    // Check output
    const output = result.stdout.toString();
    const exitCode = result.exitCode;

    if (exitCode !== 0 || output.includes('invalid') || output.includes('error')) {
      errors.push('Flat-Folder validation failed');
    }

    // Clean up
    await $`rm -f ${tmpFile}`.quiet();

  } catch (error) {
    errors.push(`Flat-Folder validation error: ${error}`);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
