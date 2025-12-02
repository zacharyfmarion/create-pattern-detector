#!/usr/bin/env bun
/**
 * Analyze crease complexity distribution of scraped patterns
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

// Read all scraped FOLD files and count their creases
const scrapedDir = '../output/scraped/validated/tier-a';
const files = readdirSync(scrapedDir).filter(f => f.endsWith('.fold'));

const creaseCounts: number[] = [];

for (const file of files) {
  try {
    const fold = JSON.parse(readFileSync(join(scrapedDir, file), 'utf8'));
    const numCreases = fold.edges_assignment
      ? fold.edges_assignment.filter((a: string) => a === 'M' || a === 'V').length
      : 0;
    creaseCounts.push(numCreases);
  } catch (e) {
    // Skip invalid files
  }
}

creaseCounts.sort((a, b) => a - b);

console.log('Scraped Pattern Crease Statistics:');
console.log('=====================================');
console.log(`Total patterns analyzed: ${creaseCounts.length}`);
console.log(`Min creases: ${Math.min(...creaseCounts)}`);
console.log(`Max creases: ${Math.max(...creaseCounts)}`);
console.log(`Mean: ${(creaseCounts.reduce((a, b) => a + b, 0) / creaseCounts.length).toFixed(1)}`);
console.log(`Median: ${creaseCounts[Math.floor(creaseCounts.length / 2)]}`);
console.log(`25th percentile: ${creaseCounts[Math.floor(creaseCounts.length * 0.25)]}`);
console.log(`75th percentile: ${creaseCounts[Math.floor(creaseCounts.length * 0.75)]}`);
console.log(`90th percentile: ${creaseCounts[Math.floor(creaseCounts.length * 0.90)]}`);
console.log(`95th percentile: ${creaseCounts[Math.floor(creaseCounts.length * 0.95)]}`);

// Distribution buckets
const buckets: Record<string, number> = {
  '0-50': 0,
  '51-100': 0,
  '101-150': 0,
  '151-200': 0,
  '201-300': 0,
  '301-500': 0,
  '500+': 0
};

for (const count of creaseCounts) {
  if (count <= 50) buckets['0-50']++;
  else if (count <= 100) buckets['51-100']++;
  else if (count <= 150) buckets['101-150']++;
  else if (count <= 200) buckets['151-200']++;
  else if (count <= 300) buckets['201-300']++;
  else if (count <= 500) buckets['301-500']++;
  else buckets['500+']++;
}

console.log('\nDistribution:');
for (const [bucket, count] of Object.entries(buckets)) {
  const pct = ((count / creaseCounts.length) * 100).toFixed(1);
  console.log(`  ${bucket.padEnd(10)} ${count.toString().padStart(4)} (${pct}%)`);
}
