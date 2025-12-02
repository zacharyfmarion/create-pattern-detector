#!/usr/bin/env bun
/**
 * Analyze why patterns are being rejected, grouped by crease count
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

interface RejectionReason {
  creaseCount: number;
  failedChecks: string[];
  passedChecks: string[];
}

function getCreaseCount(fold: any): number {
  if (!fold.edges_assignment) return 0;
  return fold.edges_assignment.filter((a: string) => a === 'M' || a === 'V').length;
}

function getBucket(creaseCount: number): string {
  if (creaseCount <= 50) return '0-50';
  if (creaseCount <= 100) return '51-100';
  if (creaseCount <= 150) return '101-150';
  if (creaseCount <= 200) return '151-200';
  if (creaseCount <= 300) return '201-300';
  if (creaseCount <= 500) return '301-500';
  return '500+';
}

// Analyze rejected patterns
const rejectedDir = '../output/synthetic/raw/rejected';
const files = readdirSync(rejectedDir).filter(f => f.endsWith('.json'));

const rejectionsByBucket: Record<string, { count: number; reasons: Record<string, number> }> = {};
const rejections: RejectionReason[] = [];

for (const file of files.slice(0, 1000)) { // Sample first 1000
  try {
    const metadata = JSON.parse(readFileSync(join(rejectedDir, file), 'utf8'));
    const fold = metadata.fold;
    const validation = metadata.validation;

    const creaseCount = getCreaseCount(fold);
    const bucket = getBucket(creaseCount);

    if (!rejectionsByBucket[bucket]) {
      rejectionsByBucket[bucket] = { count: 0, reasons: {} };
    }

    rejectionsByBucket[bucket].count++;

    // Track which checks failed
    if (validation.failed) {
      for (const failedCheck of validation.failed) {
        if (!rejectionsByBucket[bucket].reasons[failedCheck]) {
          rejectionsByBucket[bucket].reasons[failedCheck] = 0;
        }
        rejectionsByBucket[bucket].reasons[failedCheck]++;
      }
    }

    rejections.push({
      creaseCount,
      failedChecks: validation.failed || [],
      passedChecks: validation.passed || [],
    });
  } catch (e) {
    // Skip invalid files
  }
}

console.log('Rejection Analysis by Complexity Bucket');
console.log('========================================\n');

const bucketOrder = ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500', '500+'];

for (const bucket of bucketOrder) {
  const data = rejectionsByBucket[bucket];
  if (!data) continue;

  console.log(`Bucket ${bucket}: ${data.count} rejections`);

  // Sort reasons by frequency
  const sortedReasons = Object.entries(data.reasons)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5); // Top 5 reasons

  for (const [reason, count] of sortedReasons) {
    const pct = ((count / data.count) * 100).toFixed(1);
    console.log(`  ${reason.padEnd(30)} ${count.toString().padStart(4)} (${pct}%)`);
  }
  console.log();
}

// Calculate acceptance rate by crease count range
console.log('\nAcceptance Rate by Crease Count:');
console.log('=================================\n');

const tierADir = '../output/synthetic/raw/tier-a';
const acceptedFiles = readdirSync(tierADir).filter(f => f.endsWith('.fold'));

const acceptedByBucket: Record<string, number> = {};
for (const file of acceptedFiles) {
  try {
    const fold = JSON.parse(readFileSync(join(tierADir, file), 'utf8'));
    const creaseCount = getCreaseCount(fold);
    const bucket = getBucket(creaseCount);
    acceptedByBucket[bucket] = (acceptedByBucket[bucket] || 0) + 1;
  } catch (e) {
    // Skip
  }
}

for (const bucket of bucketOrder) {
  const accepted = acceptedByBucket[bucket] || 0;
  const rejected = rejectionsByBucket[bucket]?.count || 0;
  const total = accepted + rejected;

  if (total === 0) continue;

  const acceptanceRate = ((accepted / total) * 100).toFixed(1);
  console.log(`${bucket.padEnd(10)} Accepted: ${accepted.toString().padStart(5)} | Rejected: ${rejected.toString().padStart(5)} | Rate: ${acceptanceRate}%`);
}
