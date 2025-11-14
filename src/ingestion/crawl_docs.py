#!/usr/bin/env python3
"""
CLI tool for crawling design system documentation.

Usage:
    # Add a documentation source
    python crawl_docs.py add-source "Material UI" https://mui.com/material-ui/

    # Run the crawler
    python crawl_docs.py crawl

    # List configured sources
    python crawl_docs.py list-sources

    # Force re-crawl all pages
    python crawl_docs.py crawl --force
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from web_crawler import IncrementalDocCrawler


def load_config(config_path: str = "config/crawler_config.yaml") -> dict:
    """Load crawler configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str = "config/crawler_config.yaml"):
    """Save crawler configuration."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def add_source(args):
    """Add a new documentation source to crawl."""
    config = load_config()

    # Parse domain from URL
    from urllib.parse import urlparse
    parsed = urlparse(args.url)
    domain = parsed.netloc

    # Create source configuration
    new_source = {
        'name': args.name,
        'start_url': args.url,
        'allowed_domains': [domain],
        'url_patterns': [f"*{parsed.path}*"] if parsed.path != '/' else [f"*"],
        'max_depth': args.max_depth,
    }

    # Add exclude patterns if provided
    if args.exclude:
        new_source['exclude_patterns'] = args.exclude.split(',')

    # Initialize sources list if not exists
    if 'sources' not in config or config['sources'] is None:
        config['sources'] = []

    # Check if source already exists
    existing = [s for s in config['sources'] if s['name'] == args.name]
    if existing:
        print(f"⚠ Source '{args.name}' already exists. Updating...")
        config['sources'] = [s for s in config['sources'] if s['name'] != args.name]

    # Add new source
    config['sources'].append(new_source)
    save_config(config)

    print(f"✓ Added source: {args.name}")
    print(f"  URL: {args.url}")
    print(f"  Domain: {domain}")
    print(f"  Max depth: {args.max_depth}")


def list_sources(args):
    """List all configured documentation sources."""
    config = load_config()
    sources = config.get('sources', [])

    if not sources:
        print("No sources configured yet.")
        print("\nAdd a source with:")
        print('  python crawl_docs.py add-source "Name" https://docs.example.com/')
        return

    print(f"Configured documentation sources ({len(sources)}):\n")
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['name']}")
        print(f"   URL: {source['start_url']}")
        print(f"   Domains: {', '.join(source['allowed_domains'])}")
        print(f"   Max depth: {source.get('max_depth', 'unlimited')}")
        if 'exclude_patterns' in source:
            print(f"   Exclude: {', '.join(source['exclude_patterns'])}")
        print()


def crawl(args):
    """Run the crawler."""
    config = load_config()
    sources = config.get('sources', [])

    if not sources:
        print("❌ No sources configured.")
        print("\nAdd a source first:")
        print('  python crawl_docs.py add-source "Name" https://docs.example.com/')
        sys.exit(1)

    # Disable incremental crawling if force flag is set
    if args.force:
        print("⚠ Force mode: ignoring previous crawl state")
        config['incremental']['enabled'] = False
        # Temporarily save modified config
        save_config(config)

    print(f"Starting crawl of {len(sources)} source(s)...\n")

    try:
        crawler = IncrementalDocCrawler()
        crawler.run()
        print("\n✓ Crawl completed successfully!")
        print(f"\nCrawled content saved to: {config['output']['raw_output_dir']}")

    except Exception as e:
        print(f"\n❌ Crawl failed: {e}")
        sys.exit(1)

    finally:
        # Restore incremental crawling setting if it was disabled
        if args.force:
            config['incremental']['enabled'] = True
            save_config(config)


def remove_source(args):
    """Remove a documentation source."""
    config = load_config()

    if 'sources' not in config or not config['sources']:
        print("No sources configured.")
        return

    # Remove by name
    original_count = len(config['sources'])
    config['sources'] = [s for s in config['sources'] if s['name'] != args.name]

    if len(config['sources']) < original_count:
        save_config(config)
        print(f"✓ Removed source: {args.name}")
    else:
        print(f"⚠ Source not found: {args.name}")


def clear_state(args):
    """Clear crawler state to force full re-crawl."""
    config = load_config()
    state_path = Path(config['incremental']['state_storage'])

    if state_path.exists():
        if args.yes or input("Clear all crawl state? This will re-crawl everything. (y/N): ").lower() == 'y':
            state_path.unlink()
            print("✓ Crawler state cleared")
        else:
            print("Cancelled")
    else:
        print("No crawler state found")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Crawl design system documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add documentation sources
  python crawl_docs.py add-source "Material UI" https://mui.com/material-ui/
  python crawl_docs.py add-source "Ant Design" https://ant.design/components/

  # List sources
  python crawl_docs.py list-sources

  # Run crawler (incremental - only new/changed pages)
  python crawl_docs.py crawl

  # Force full re-crawl
  python crawl_docs.py crawl --force
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Add source command
    add_parser = subparsers.add_parser('add-source', help='Add a documentation source')
    add_parser.add_argument('name', help='Name of the documentation source')
    add_parser.add_argument('url', help='Starting URL to crawl')
    add_parser.add_argument('--max-depth', type=int, default=3, help='Maximum crawl depth (default: 3)')
    add_parser.add_argument('--exclude', help='Comma-separated URL patterns to exclude')
    add_parser.set_defaults(func=add_source)

    # List sources command
    list_parser = subparsers.add_parser('list-sources', help='List configured sources')
    list_parser.set_defaults(func=list_sources)

    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Run the crawler')
    crawl_parser.add_argument('--force', action='store_true', help='Force re-crawl all pages')
    crawl_parser.set_defaults(func=crawl)

    # Remove source command
    remove_parser = subparsers.add_parser('remove-source', help='Remove a documentation source')
    remove_parser.add_argument('name', help='Name of the source to remove')
    remove_parser.set_defaults(func=remove_source)

    # Clear state command
    clear_parser = subparsers.add_parser('clear-state', help='Clear crawler state')
    clear_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    clear_parser.set_defaults(func=clear_state)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
