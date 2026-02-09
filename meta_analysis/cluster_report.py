"""
Generate a detailed summary report for LLM-clustered categories.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def build_category_lookup(categorization_data: dict) -> dict:
    """
    Build a lookup dict from category name to category data.
    
    Returns:
        Dict mapping category_name -> {definition, examples, num_examples}
    """
    lookup = {}
    categories = categorization_data.get("categorization", {}).get("categories", [])
    
    for cat in categories:
        name = cat["category_name"]
        lookup[name] = {
            "definition": cat.get("definition", ""),
            "examples": cat.get("examples", []),
            "num_examples": len(cat.get("examples", []))
        }
    
    return lookup


def generate_report(
    cluster_path: str,
    categorization_path: str = None,
    output_path: str = None,
    max_examples_per_category: int = 3,
) -> str:
    """
    Generate a detailed summary report for clustered categories.
    
    Args:
        cluster_path: Path to the LLM clusters JSON file
        categorization_path: Path to the original categorization JSON file (optional, 
                            will be read from cluster file metadata if not provided)
        output_path: Optional path to save the report (defaults to cluster_report_{timestamp}.md)
        max_examples_per_category: Max examples to show per category
        
    Returns:
        The generated report as a string
    """
    # Load cluster data
    cluster_data = load_json(cluster_path)
    
    # Get categorization path from cluster metadata if not provided
    if categorization_path is None:
        categorization_path = cluster_data.get("metadata", {}).get("input_path")
        if not categorization_path:
            raise ValueError("No categorization_path provided and could not find 'input_path' in cluster file metadata")
        print(f"Using categorization file from cluster metadata: {categorization_path}")
    
    categorization_data = load_json(categorization_path)
    
    # Build category lookup
    category_lookup = build_category_lookup(categorization_data)
    
    # Calculate total examples across all categories
    total_examples = sum(cat["num_examples"] for cat in category_lookup.values())
    total_categories = len(category_lookup)
    
    clusters = cluster_data.get("clusters", [])
    
    # Build report
    lines = []
    lines.append("# Cluster Summary Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Cluster File:** `{Path(cluster_path).name}`")
    lines.append(f"**Categorization File:** `{Path(categorization_path).name}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overall statistics
    lines.append("## Overall Statistics")
    lines.append("")
    lines.append(f"- **Total Categories (Original):** {total_categories}")
    lines.append(f"- **Total Successful Perturbations:** {total_examples}")
    lines.append(f"- **Number of Clusters:** {len(clusters)}")
    lines.append("")
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster in clusters:
        cluster_name = cluster.get("cluster_name", "Unnamed")
        member_categories = cluster.get("member_categories", [])
        
        # Count examples for each member category
        cluster_examples = 0
        category_details = []
        
        for member in member_categories:
            member_name = member.get("category_name", "")
            cat_data = category_lookup.get(member_name, {})
            num_examples = cat_data.get("num_examples", 0)
            cluster_examples += num_examples
            category_details.append({
                "name": member_name,
                "num_examples": num_examples,
                "examples": cat_data.get("examples", []),
                "justification": member.get("justification", "")
            })
        
        proportion = (cluster_examples / total_examples * 100) if total_examples > 0 else 0
        
        cluster_stats.append({
            "cluster_name": cluster_name,
            "definition": cluster.get("definition", ""),
            "anchor_phrases": cluster.get("anchor_phrases", ""),
            "num_categories": len(member_categories),
            "total_examples": cluster_examples,
            "proportion": proportion,
            "categories": category_details
        })
    
    # Sort by total examples (descending)
    cluster_stats.sort(key=lambda x: -x["total_examples"])
    
    # Summary table
    lines.append("## Cluster Summary Table")
    lines.append("")
    lines.append("| Rank | Cluster | Categories | Perturbations | % of Total |")
    lines.append("|------|---------|------------|---------------|------------|")
    
    for i, cs in enumerate(cluster_stats, 1):
        # Truncate cluster name for table
        name_short = cs["cluster_name"][:50] + "..." if len(cs["cluster_name"]) > 50 else cs["cluster_name"]
        lines.append(f"| {i} | {name_short} | {cs['num_categories']} | {cs['total_examples']} | {cs['proportion']:.1f}% |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed cluster breakdown
    lines.append("## Detailed Cluster Breakdown")
    lines.append("")
    
    for i, cs in enumerate(cluster_stats, 1):
        lines.append(f"### {i}. {cs['cluster_name']}")
        lines.append("")
        lines.append(f"**Definition:** {cs['definition']}")
        lines.append("")
        lines.append(f"**Anchor Phrases:** {cs['anchor_phrases']}")
        lines.append("")
        lines.append(f"**Statistics:**")
        lines.append(f"- Member Categories: {cs['num_categories']}")
        lines.append(f"- Total Perturbations: {cs['total_examples']}")
        lines.append(f"- Proportion of Dataset: {cs['proportion']:.1f}%")
        lines.append("")
        
        lines.append("#### Member Categories")
        lines.append("")
        
        # Sort categories by example count
        sorted_categories = sorted(cs["categories"], key=lambda x: -x["num_examples"])
        
        for cat in sorted_categories:
            lines.append(f"**{cat['name']}** ({cat['num_examples']} perturbations)")
            lines.append("")
            if cat["justification"]:
                lines.append(f"*Cluster Justification:* {cat['justification']}")
                lines.append("")
            
            # Show example trigger phrases
            if cat["examples"]:
                lines.append("*Example Trigger Phrases:*")
                for j, ex in enumerate(cat["examples"][:max_examples_per_category], 1):
                    trigger = ex.get("trigger_phrase", "N/A")
                    lines.append(f"  {j}. \"{trigger}\"")
                
                if len(cat["examples"]) > max_examples_per_category:
                    lines.append(f"  ... and {len(cat['examples']) - max_examples_per_category} more")
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Coverage analysis
    lines.append("## Coverage Analysis")
    lines.append("")
    
    # Check for unclustered categories
    clustered_category_names = set()
    for cluster in clusters:
        for member in cluster.get("member_categories", []):
            clustered_category_names.add(member.get("category_name", ""))
    
    unclustered = set(category_lookup.keys()) - clustered_category_names
    
    if unclustered:
        lines.append(f"**Unclustered Categories ({len(unclustered)}):**")
        for cat_name in sorted(unclustered):
            cat_data = category_lookup[cat_name]
            lines.append(f"- {cat_name} ({cat_data['num_examples']} perturbations)")
        lines.append("")
    else:
        lines.append("All categories are included in clusters.")
        lines.append("")
    
    # Cumulative coverage
    lines.append("### Cumulative Coverage")
    lines.append("")
    lines.append("| Top N Clusters | Categories | Perturbations | Cumulative % |")
    lines.append("|----------------|------------|---------------|--------------|")
    
    cumulative_examples = 0
    cumulative_categories = 0
    for i, cs in enumerate(cluster_stats, 1):
        cumulative_examples += cs["total_examples"]
        cumulative_categories += cs["num_categories"]
        cumulative_pct = (cumulative_examples / total_examples * 100) if total_examples > 0 else 0
        lines.append(f"| {i} | {cumulative_categories} | {cumulative_examples} | {cumulative_pct:.1f}% |")
    
    lines.append("")
    
    report = "\n".join(lines)
    
    # Save report
    if output_path is None:
        cluster_dir = Path(cluster_path).parent
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = cluster_dir / f"cluster_report_{timestamp}.md"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a detailed summary report for LLM-clustered categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Uses categorization path from cluster file metadata automatically
    python cluster_report.py --cluster_path "path/to/llm_clusters.json"

    # Override with explicit categorization path
    python cluster_report.py \\
        --cluster_path "path/to/llm_clusters.json" \\
        --categorization_path "path/to/categorization.json"
        """
    )

    parser.add_argument('--cluster_path', type=str, required=True,
                       help='Path to the LLM clusters JSON file')
    parser.add_argument('--categorization_path', type=str, default=None,
                       help='Path to the original categorization JSON file (optional, read from cluster metadata if not provided)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for the report (defaults to cluster_report_{timestamp}.md)')
    parser.add_argument('--max_examples', type=int, default=3,
                       help='Max trigger phrase examples to show per category (default: 3)')
    parser.add_argument('--print_report', action='store_true',
                       help='Print the report to stdout')

    return parser.parse_args()


def main():
    args = parse_args()
    
    report = generate_report(
        cluster_path=args.cluster_path,
        categorization_path=args.categorization_path,
        output_path=args.output_path,
        max_examples_per_category=args.max_examples
    )
    
    if args.print_report:
        print(report)


if __name__ == "__main__":
    main()
