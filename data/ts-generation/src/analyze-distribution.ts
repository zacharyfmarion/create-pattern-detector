#!/usr/bin/env bun
/**
 * Analyze complexity distribution of synthetic patterns and compare to scraped targets
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

// Target distribution from scraped patterns
const TARGET_DISTRIBUTION = {
  '30-50': 22.6,
  '51-100': 5.2,
  '101-150': 14.8,
  '151-200': 11.6,
  '201-300': 20.0,
  '301-500': 5.8,
  '500-800': 20.0,  // Capped from 500+
};

const TARGET_COUNT = 12000; // Total accepted patterns we want

interface BucketStats {
  count: number;
  percentage: number;
  targetCount: number;
  targetPercentage: number;
  difference: number;
}

function getCreaseCount(fold: any): number {
  if (!fold.edges_assignment) return 0;
  return fold.edges_assignment.filter((a: string) => a === 'M' || a === 'V').length;
}

function getBucket(creaseCount: number): string | null {
  if (creaseCount >= 30 && creaseCount <= 50) return '30-50';
  if (creaseCount >= 51 && creaseCount <= 100) return '51-100';
  if (creaseCount >= 101 && creaseCount <= 150) return '101-150';
  if (creaseCount >= 151 && creaseCount <= 200) return '151-200';
  if (creaseCount >= 201 && creaseCount <= 300) return '201-300';
  if (creaseCount >= 301 && creaseCount <= 500) return '301-500';
  if (creaseCount >= 501 && creaseCount <= 800) return '500-800';
  return null; // Outside our range
}

// Analyze accepted patterns
// Path is relative to cwd (data/ts-generation), so ../output goes to data/output
const tierADir = '../output/synthetic/raw/tier-a';
const files = readdirSync(tierADir).filter(f => f.endsWith('.fold'));

const bucketCounts: Record<string, number> = {
  '30-50': 0,
  '51-100': 0,
  '101-150': 0,
  '151-200': 0,
  '201-300': 0,
  '301-500': 0,
  '500-800': 0,
};

const creaseCounts: number[] = [];

for (const file of files) {
  try {
    const fold = JSON.parse(readFileSync(join(tierADir, file), 'utf8'));
    const numCreases = getCreaseCount(fold);
    creaseCounts.push(numCreases);

    const bucket = getBucket(numCreases);
    if (bucket) {
      bucketCounts[bucket]++;
    }
  } catch (e) {
    // Skip invalid files
  }
}

const totalAccepted = creaseCounts.length;

console.log('Synthetic Pattern Analysis (Accepted Tier-A Only)');
console.log('=================================================');
console.log(`Total accepted patterns: ${totalAccepted}`);
console.log(`Min creases: ${Math.min(...creaseCounts)}`);
console.log(`Max creases: ${Math.max(...creaseCounts)}`);
console.log(`Mean: ${(creaseCounts.reduce((a, b) => a + b, 0) / creaseCounts.length).toFixed(1)}`);
console.log(`Median: ${creaseCounts[Math.floor(creaseCounts.length / 2)]}`);

console.log('\nDistribution by Bucket:');
console.log('=======================\n');

const bucketStats: Record<string, BucketStats> = {};

for (const [bucket, count] of Object.entries(bucketCounts)) {
  const percentage = (count / totalAccepted) * 100;
  const targetPercentage = TARGET_DISTRIBUTION[bucket as keyof typeof TARGET_DISTRIBUTION];
  const targetCount = Math.round((targetPercentage / 100) * TARGET_COUNT);
  const difference = percentage - targetPercentage;

  bucketStats[bucket] = {
    count,
    percentage,
    targetCount,
    targetPercentage,
    difference,
  };

  console.log(`Bucket ${bucket}:`);
  console.log(`  Current:  ${count.toString().padStart(5)} patterns (${percentage.toFixed(1)}%)`);
  console.log(`  Target:   ${targetCount.toString().padStart(5)} patterns (${targetPercentage.toFixed(1)}%)`);
  console.log(`  Gap:      ${(targetCount - count).toString().padStart(5)} patterns (${difference > 0 ? '+' : ''}${difference.toFixed(1)}%)`);
  console.log();
}

// Calculate estimated acceptance rates (from 15K generated to current accepted)
// Note: This is an overall estimate - we can't easily determine per-bucket rejection
// from this data alone without tracking during generation
const overallAcceptanceRate = totalAccepted / 15000;
console.log('\nOverall Statistics:');
console.log('==================');
console.log(`Overall acceptance rate: ${(overallAcceptanceRate * 100).toFixed(1)}%`);
console.log(`Patterns generated: 15,000`);
console.log(`Patterns accepted: ${totalAccepted}`);
console.log(`Patterns rejected: ${15000 - totalAccepted}`);

// Recommended generation strategy
console.log('\n\nRecommended Bucket-Based Generation:');
console.log('====================================\n');
console.log('Assuming similar acceptance rates per bucket (~62%), generate:\n');

for (const [bucket, stats] of Object.entries(bucketStats)) {
  const needed = stats.targetCount;
  const estimatedToGenerate = Math.ceil(needed / overallAcceptanceRate);
  console.log(`Bucket ${bucket}:`);
  console.log(`  Target:    ${needed} accepted patterns`);
  console.log(`  Generate:  ~${estimatedToGenerate} patterns (accounting for ~${(overallAcceptanceRate * 100).toFixed(1)}% acceptance)`);
  console.log();
}

console.log('⚠️  Note: Complex patterns may have LOWER acceptance rates.');
console.log('   Monitor actual acceptance during generation and adjust counts accordingly.');
console.log('   Start with simpler buckets to measure actual rates, then adjust for complex buckets.\n');
