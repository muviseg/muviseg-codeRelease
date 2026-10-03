"""
Evaluation Metrics for Segment Matching

Computes AUPRC, R@1, and R@5 metrics for segment matching evaluation.
"""

import numpy as np
from sklearn.metrics import average_precision_score
from typing import Dict, List, Tuple
import warnings


def compute_metrics(scores: np.ndarray, gt_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute AUPRC, R@1, and R@5 metrics for segment matching.

    Args:
        scores: (M, N) matching scores from model (higher = better match)
        gt_labels: (M, N) binary ground truth correspondences (1 = match, 0 = no match)

    Returns:
        Dictionary with keys:
            - 'AUPRC': Mean average precision across all queries
            - 'R@1': Recall @ top-1 (fraction of queries where top-1 match is correct)
            - 'R@5': Recall @ top-5 (fraction of queries where any top-5 match is correct)
            - 'num_queries': Number of valid queries (queries with at least one positive)
    """
    M, N = scores.shape
    assert gt_labels.shape == (M, N), f"Shape mismatch: scores {scores.shape} vs gt {gt_labels.shape}"

    auprc_list = []
    recall_at_1_list = []
    recall_at_5_list = []

    # Process each reference segment (query)
    for i in range(M):
        query_scores = scores[i]  # (N,)
        query_gt = gt_labels[i]  # (N,)

        # Skip if no positive matches for this query
        num_positives = query_gt.sum()
        if num_positives == 0:
            continue

        # Compute AUPRC
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            ap = average_precision_score(query_gt, query_scores)
        auprc_list.append(ap)

        # Compute Recall@1
        top1_idx = np.argmax(query_scores)
        recall_at_1 = float(query_gt[top1_idx])
        recall_at_1_list.append(recall_at_1)

        # Compute Recall@5
        # Get top-5 indices
        k = min(5, N)
        top5_indices = np.argpartition(query_scores, -k)[-k:]

        # Check if any of top-5 is a positive match
        recall_at_5 = float(np.any(query_gt[top5_indices] == 1))
        recall_at_5_list.append(recall_at_5)

    # Compute mean metrics
    num_queries = len(auprc_list)

    if num_queries == 0:
        return {
            'AUPRC': 0.0,
            'R@1': 0.0,
            'R@5': 0.0,
            'num_queries': 0
        }

    return {
        'AUPRC': float(np.mean(auprc_list)),
        'R@1': float(np.mean(recall_at_1_list)),
        'R@5': float(np.mean(recall_at_5_list)),
        'num_queries': num_queries
    }


def aggregate_metrics_by_bin(
    results_by_bin: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple batches for each pose bin.

    Args:
        results_by_bin: Dictionary mapping bin name to list of metric dicts from batches

    Returns:
        Dictionary mapping bin name to aggregated metrics
    """
    aggregated = {}

    for bin_name, metrics_list in results_by_bin.items():
        if len(metrics_list) == 0:
            aggregated[bin_name] = {
                'AUPRC': 0.0,
                'R@1': 0.0,
                'R@5': 0.0,
                'num_queries': 0
            }
            continue

        # Collect all AUPRC, R@1, R@5 values weighted by num_queries
        total_queries = sum(m['num_queries'] for m in metrics_list)

        if total_queries == 0:
            aggregated[bin_name] = {
                'AUPRC': 0.0,
                'R@1': 0.0,
                'R@5': 0.0,
                'num_queries': 0
            }
            continue

        # Weighted average
        auprc_weighted = sum(m['AUPRC'] * m['num_queries'] for m in metrics_list) / total_queries
        r1_weighted = sum(m['R@1'] * m['num_queries'] for m in metrics_list) / total_queries
        r5_weighted = sum(m['R@5'] * m['num_queries'] for m in metrics_list) / total_queries

        aggregated[bin_name] = {
            'AUPRC': float(auprc_weighted),
            'R@1': float(r1_weighted),
            'R@5': float(r5_weighted),
            'num_queries': total_queries
        }

    return aggregated


def print_table2_format(results_by_bin: Dict[str, Dict[str, float]]) -> str:
    """
    Print results in Table 2 format matching the paper.

    Args:
        results_by_bin: Dictionary mapping bin name (e.g., "0-45") to metrics

    Returns:
        Formatted table as string
    """
    # Auto-detect bin order from results (sort by first number in "X-Y" format)
    def parse_bin_name(bin_name):
        """Extract start angle from bin name like '0-45' -> 0"""
        try:
            return int(bin_name.split('-')[0])
        except:
            return 0

    bin_order = sorted(results_by_bin.keys(), key=parse_bin_name)

    # Table header
    lines = []
    lines.append("=" * 80)
    lines.append("Table 2: Segment Matching Results")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Pose Bin':>12} | {'AUPRC':>8} | {'R@1':>8} | {'R@5':>8} | {'#Queries':>10}")
    lines.append("-" * 80)

    # Print metrics for each bin
    for bin_name in bin_order:
        if bin_name not in results_by_bin:
            continue

        metrics = results_by_bin[bin_name]
        auprc = metrics['AUPRC'] * 100  # Convert to percentage
        r1 = metrics['R@1'] * 100
        r5 = metrics['R@5'] * 100
        num_q = metrics['num_queries']

        lines.append(
            f"{bin_name:>12} | {auprc:>8.2f} | {r1:>8.2f} | {r5:>8.2f} | {num_q:>10}"
        )

    lines.append("=" * 80)

    # Compute overall average (weighted by number of queries)
    total_queries = sum(m['num_queries'] for m in results_by_bin.values())
    if total_queries > 0:
        avg_auprc = sum(m['AUPRC'] * m['num_queries'] for m in results_by_bin.values()) / total_queries * 100
        avg_r1 = sum(m['R@1'] * m['num_queries'] for m in results_by_bin.values()) / total_queries * 100
        avg_r5 = sum(m['R@5'] * m['num_queries'] for m in results_by_bin.values()) / total_queries * 100

        lines.append(f"{'Overall':>12} | {avg_auprc:>8.2f} | {avg_r1:>8.2f} | {avg_r5:>8.2f} | {total_queries:>10}")
        lines.append("=" * 80)

    return "\n".join(lines)


def compute_batch_metrics(
    scores_batch: List[np.ndarray],
    gt_batch: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute metrics for a batch of score/gt pairs.

    Args:
        scores_batch: List of score matrices, each (M_i, N_i)
        gt_batch: List of ground truth matrices, each (M_i, N_i)

    Returns:
        Aggregated metrics across the batch
    """
    all_auprc = []
    all_r1 = []
    all_r5 = []
    total_queries = 0

    for scores, gt in zip(scores_batch, gt_batch):
        metrics = compute_metrics(scores, gt)
        if metrics['num_queries'] > 0:
            # Weight by number of queries
            all_auprc.append(metrics['AUPRC'] * metrics['num_queries'])
            all_r1.append(metrics['R@1'] * metrics['num_queries'])
            all_r5.append(metrics['R@5'] * metrics['num_queries'])
            total_queries += metrics['num_queries']

    if total_queries == 0:
        return {
            'AUPRC': 0.0,
            'R@1': 0.0,
            'R@5': 0.0,
            'num_queries': 0
        }

    return {
        'AUPRC': float(sum(all_auprc) / total_queries),
        'R@1': float(sum(all_r1) / total_queries),
        'R@5': float(sum(all_r5) / total_queries),
        'num_queries': total_queries
    }


if __name__ == "__main__":
    # Quick test
    print("Testing evaluation metrics...")

    # Synthetic test case
    np.random.seed(42)
    M, N = 10, 15

    # Create some scores
    scores = np.random.rand(M, N)

    # Create ground truth with some positive matches
    gt = np.zeros((M, N), dtype=np.uint8)
    gt[0, 2] = 1  # Query 0 matches target 2
    gt[1, 5] = 1  # Query 1 matches target 5
    gt[2, 5] = 1  # Query 2 also matches target 5
    gt[3, 10] = 1  # Query 3 matches target 10

    # Make scores higher for true matches
    for i in range(M):
        for j in range(N):
            if gt[i, j] == 1:
                scores[i, j] = 0.9 + np.random.rand() * 0.1

    metrics = compute_metrics(scores, gt)
    print(f"\nMetrics:")
    print(f"  AUPRC: {metrics['AUPRC']:.4f}")
    print(f"  R@1: {metrics['R@1']:.4f}")
    print(f"  R@5: {metrics['R@5']:.4f}")
    print(f"  Num queries: {metrics['num_queries']}")

    # Test aggregation
    results_by_bin = {
        "0-45": [
            {'AUPRC': 0.95, 'R@1': 0.96, 'R@5': 0.98, 'num_queries': 100},
            {'AUPRC': 0.94, 'R@1': 0.95, 'R@5': 0.97, 'num_queries': 100}
        ],
        "45-90": [
            {'AUPRC': 0.86, 'R@1': 0.91, 'R@5': 0.96, 'num_queries': 100}
        ]
    }

    aggregated = aggregate_metrics_by_bin(results_by_bin)
    print("\n" + print_table2_format(aggregated))
