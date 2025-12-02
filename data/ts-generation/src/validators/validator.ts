/**
 * Orchestrator for all validation checks
 *
 * Implements tiered validation:
 * - Tier S (Gold): Passes all checks (local + global)
 * - Tier A (Silver): Passes local checks only
 * - REJECT: Fails local checks
 */

import type { FOLDFormat, ValidationResult, ValidationTier } from '../types/crease-pattern.ts';
import { validateLocal } from './local.ts';
import { validateGlobal, validateFlatFolder } from './global.ts';

/**
 * Full validation with tiered classification
 *
 * @param fold FOLD format crease pattern
 * @param options Validation options
 * @returns ValidationResult with tier classification
 */
export async function validate(
  fold: FOLDFormat,
  options: {
    skipGlobal?: boolean;
    skipFlatFolder?: boolean;
  } = {}
): Promise<ValidationResult> {
  // Step 1: Local validation (fast)
  const localResult = validateLocal(fold);

  // If local validation fails, immediately reject
  if (localResult.tier === 'REJECT') {
    return localResult;
  }

  // Step 2: Global validation (slower)
  if (!options.skipGlobal) {
    try {
      const globalResult = await validateGlobal(fold);

      if (globalResult.valid) {
        localResult.passed.push('fold-cli-flat-fold');
      } else {
        localResult.failed.push('fold-cli-flat-fold');
        localResult.errors.push(...globalResult.errors);
      }

      // Step 3: Flat-Folder validation (optional)
      if (!options.skipFlatFolder) {
        const flatFolderResult = await validateFlatFolder(fold);

        if (flatFolderResult.valid) {
          localResult.passed.push('flat-folder');
        } else {
          localResult.failed.push('flat-folder');
          localResult.errors.push(...flatFolderResult.errors);
        }
      }
    } catch (error) {
      localResult.errors.push(`Global validation error: ${error}`);
      localResult.failed.push('global-validation');
    }
  }

  // Determine final tier
  const tier = determineTier(localResult.passed, localResult.failed);
  localResult.tier = tier;

  return localResult;
}

/**
 * Determine validation tier based on passed/failed checks
 *
 * Tier S: Passes all local checks + global checks
 * Tier A: Passes all local checks only
 * REJECT: Fails any local check
 */
function determineTier(passed: string[], failed: string[]): ValidationTier {
  // Required local checks
  const requiredLocal = [
    'maekawa',
    'kawasaki',
    'no-self-intersections',
    'complete-border',
    '2-colorable',
  ];

  // Check if all local checks passed
  const allLocalPassed = requiredLocal.every(check => passed.includes(check));

  if (!allLocalPassed) {
    return 'REJECT';
  }

  // Check if global checks passed
  const globalPassed = passed.includes('fold-cli-flat-fold');

  if (globalPassed) {
    return 'S'; // Gold tier
  } else {
    return 'A'; // Silver tier
  }
}

/**
 * Quick validation for generation loop (local only)
 *
 * @param fold FOLD format crease pattern
 * @returns True if passes local validation
 */
export function quickValidate(fold: FOLDFormat): boolean {
  const result = validateLocal(fold);
  return result.tier !== 'REJECT';
}

/**
 * Batch validation for multiple patterns
 *
 * @param folds Array of FOLD patterns
 * @param options Validation options
 * @returns Array of validation results
 */
export async function batchValidate(
  folds: FOLDFormat[],
  options: {
    skipGlobal?: boolean;
    skipFlatFolder?: boolean;
    parallel?: boolean;
  } = {}
): Promise<ValidationResult[]> {
  if (options.parallel) {
    return await Promise.all(
      folds.map(fold => validate(fold, options))
    );
  } else {
    const results: ValidationResult[] = [];
    for (const fold of folds) {
      results.push(await validate(fold, options));
    }
    return results;
  }
}
