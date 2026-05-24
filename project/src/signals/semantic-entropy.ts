/**
 * Signal: semantic entropy (Farquhar et al., Nature 2024).
 *
 * Reference: "Detecting Hallucinations in Large Language Models Using
 * Semantic Entropy", https://www.nature.com/articles/s41586-024-07421-0
 *
 * Mechanism. Given N completions to the same question, cluster them by
 * *meaning* rather than surface text: two samples are in the same cluster
 * iff they bidirectionally entail one another under NLI. Compute the
 * Shannon entropy of the cluster-size distribution. High semantic entropy
 * = the model is confabulating (surface-different answers, same underlying
 * uncertainty). A better signal than token-level perplexity for
 * confident-but-wrong cases because it works on meaning, not tokens.
 *
 * Why this fits Verity cleanly. The NLI cross-encoder is already loaded
 * (project/src/nli/classifier.ts) and the consistency check already
 * generates the sample stream. Semantic entropy reuses both. The signal
 * is surfaced as advisory in the rendered Markdown block: it never flips
 * the verdict.
 *
 * Cost. Quadratic in sample count: N(N-1)/2 NLI pair calls plus the same
 * again for the reverse direction. With N=5 deep-mode samples that's 20
 * pair calls at ~150 ms each on CPU = ~3 s — well inside the deep-mode
 * budget. With N=2 it's 2 calls.
 *
 * Degenerate cases.
 *   - empty samples       → entropy 0, zero clusters.
 *   - single sample       → entropy 0, one cluster.
 *   - all samples agree   → entropy 0, one cluster.
 *   - all samples differ  → entropy = log(N), N clusters.
 */

import { classifyEntailment, classifyNliLabel } from "../nli/classifier.js";

/**
 * Minimal NLI primitive needed by the entropy computer.
 *
 * The real classifier (project/src/nli/classifier.ts) exports
 * `classifyEntailment` with this exact shape. We accept it as an
 * injected dependency so the function is unit-testable without loading
 * the ~1 GB DeBERTa model.
 */
export interface NliEntailer {
  (
    premise: string,
    hypothesis: string
  ): Promise<{ label: string; score: number }>;
}

export interface SemanticEntropyResult {
  /** Shannon entropy of the cluster-size distribution, in nats. */
  entropy: number;
  /** Clusters: each is the list of sample texts that share a meaning. */
  clusters: string[][];
  /** Number of distinct meaning-clusters. Equals clusters.length. */
  clusterCount: number;
}

/**
 * Two samples are in the same cluster iff each entails the other under
 * NLI. Bidirectional entailment is the standard proxy for "same meaning"
 * used in the Farquhar et al. paper. A unidirectional entailment is too
 * weak (one side could be strictly more specific than the other).
 *
 * Returns true when the NLI top label is "entail" on BOTH directions.
 * Neutral / contradiction on either side fails the test.
 */
async function mutualEntailment(
  a: string,
  b: string,
  nli: NliEntailer
): Promise<boolean> {
  // Defensive: trim and reject empties before paying NLI.
  const aT = a.trim();
  const bT = b.trim();
  if (aT.length === 0 || bT.length === 0) return false;
  // Trivial identity short-circuit. The NLI model would almost always
  // return entail-entail on identical inputs, but skipping the call
  // saves two model invocations and matches the spirit of the cluster
  // definition (a sample is always in its own cluster).
  if (aT === bT) return true;

  try {
    const fwd = await nli(aT, bT);
    if (classifyNliLabel(fwd.label) !== "entail") return false;
    const rev = await nli(bT, aT);
    return classifyNliLabel(rev.label) === "entail";
  } catch {
    // Conservative: an NLI failure means we cannot prove same-meaning,
    // so the pair is treated as different-meaning. This biases entropy
    // upward (more clusters) on a flaky NLI runtime; preferable to the
    // alternative of silently undercounting hallucination signal.
    return false;
  }
}

/**
 * Cluster a list of sample texts by mutual-entailment NLI.
 *
 * Greedy single-pass algorithm:
 *   - for each sample, test it against the canonical (first) member of
 *     every existing cluster.
 *   - if a match is found, append to that cluster.
 *   - otherwise, start a new cluster.
 *
 * Greedy clustering is order-dependent in pathological cases (transitivity
 * is not guaranteed by NLI) but stable in practice for the small N (2-10)
 * we use. The cost is N * (number of clusters so far) NLI pair-calls,
 * worst case O(N^2) when every sample is its own cluster.
 */
async function clusterByMeaning(
  samples: string[],
  nli: NliEntailer
): Promise<string[][]> {
  const clusters: string[][] = [];
  for (const sample of samples) {
    let placed = false;
    for (const cluster of clusters) {
      // Use the first member as the canonical representative.
      if (await mutualEntailment(cluster[0]!, sample, nli)) {
        cluster.push(sample);
        placed = true;
        break;
      }
    }
    if (!placed) clusters.push([sample]);
  }
  return clusters;
}

/**
 * Shannon entropy in nats of a probability distribution `p`. `p` is
 * assumed normalised (sums to 1). Standard formula H = -sum p_i log p_i,
 * with the convention 0 log 0 = 0.
 */
function shannonEntropy(probabilities: number[]): number {
  let h = 0;
  for (const p of probabilities) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/**
 * Compute semantic entropy over a set of sample completions.
 *
 * @param samples  The model's N completions to the same prompt. Free-form
 *                 text. Empty strings are filtered.
 * @param nli      NLI entailment primitive — pass `classifyEntailment` from
 *                 project/src/nli/classifier.ts in production; a stub in
 *                 tests.
 *
 * Returns:
 *   entropy       Shannon entropy in nats over the cluster-size
 *                 distribution. 0 when all samples cluster as one meaning;
 *                 log(N) when every sample is a distinct meaning.
 *   clusters      The clustering itself, in the order clusters were formed.
 *   clusterCount  clusters.length.
 *
 * Degenerate cases:
 *   - samples.length === 0 → entropy 0, zero clusters.
 *   - samples.length === 1 → entropy 0, one cluster.
 */
export async function computeSemanticEntropy(
  samples: string[],
  nli: NliEntailer
): Promise<SemanticEntropyResult> {
  const usable = samples.filter((s) => typeof s === "string" && s.trim().length > 0);

  if (usable.length === 0) {
    return { entropy: 0, clusters: [], clusterCount: 0 };
  }
  if (usable.length === 1) {
    return { entropy: 0, clusters: [[usable[0]!]], clusterCount: 1 };
  }

  const clusters = await clusterByMeaning(usable, nli);
  const total = clusters.reduce((n, c) => n + c.length, 0);
  const probabilities = clusters.map((c) => c.length / total);
  const entropy = shannonEntropy(probabilities);

  return {
    entropy: Number(entropy.toFixed(4)),
    clusters,
    clusterCount: clusters.length,
  };
}

/**
 * Convenience wrapper: compute semantic entropy using Verity's loaded
 * DeBERTa cross-encoder (project/src/nli/classifier.ts).
 *
 * Use this from the pipeline; use computeSemanticEntropy directly with an
 * injected stub from unit tests.
 */
export async function computeSemanticEntropyWithDefaultNli(
  samples: string[]
): Promise<SemanticEntropyResult> {
  return computeSemanticEntropy(samples, classifyEntailment);
}
