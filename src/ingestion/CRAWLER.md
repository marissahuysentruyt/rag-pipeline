# Web Crawler Documentation

The RAG pipeline includes an incremental web crawler for collecting design system documentation from public websites.

## Features

- ✅ **Incremental Crawling**: Only processes new or changed pages
- ✅ **Change Detection**: Uses content hashing to detect updates
- ✅ **Multiple Sources**: Crawl documentation from multiple sites
- ✅ **Smart Content Extraction**: Removes navigation, preserves code blocks
- ✅ **Markdown Conversion**: Clean HTML to Markdown conversion
- ✅ **Metadata Extraction**: Automatic and custom metadata extraction
- ✅ **Politeness**: Respects robots.txt, configurable delays

## Quick Start

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt

# Install Playwright browsers (if needed for JavaScript-heavy sites)
playwright install
```

### 2. Add Documentation Sources

Add the URLs of design system documentation you want to crawl:

```bash
# Add a source
python src/ingestion/crawl_docs.py add-source "Material UI" https://mui.com/material-ui/getting-started/

# Add another source
python src/ingestion/crawl_docs.py add-source "Ant Design" https://ant.design/components/overview/

# List configured sources
python src/ingestion/crawl_docs.py list-sources
```

### 3. Run the Crawler

```bash
# Run crawler (incremental - only new/changed pages)
python src/ingestion/crawl_docs.py crawl

# Force full re-crawl of everything
python src/ingestion/crawl_docs.py crawl --force
```

### 4. Check Output

Crawled content is saved to `data/raw/crawled/` as Markdown files with metadata frontmatter.

## CLI Commands

### Add a Source

```bash
python src/ingestion/crawl_docs.py add-source <name> <url> [options]

Options:
  --max-depth N        Maximum crawl depth (default: 3)
  --exclude PATTERNS   Comma-separated URL patterns to exclude

Examples:
  python src/ingestion/crawl_docs.py add-source "Chakra UI" https://chakra-ui.com/docs/
  python src/ingestion/crawl_docs.py add-source "Carbon" https://carbondesignsystem.com/ --max-depth 2
```

### List Sources

```bash
python src/ingestion/crawl_docs.py list-sources
```

### Remove a Source

```bash
python src/ingestion/crawl_docs.py remove-source <name>

Example:
  python src/ingestion/crawl_docs.py remove-source "Material UI"
```

### Crawl

```bash
python src/ingestion/crawl_docs.py crawl [--force]

Options:
  --force    Ignore previous crawl state and re-crawl everything
```

### Clear State

```bash
python src/ingestion/crawl_docs.py clear-state [-y]

Options:
  -y, --yes    Skip confirmation prompt
```

## Configuration

Edit `config/crawler_config.yaml` to customize crawler behavior:

### Crawler Settings

```yaml
crawler:
  max_requests_per_crawl: 1000     # Max pages per crawl run
  max_concurrent_requests: 5        # Parallel requests
  request_timeout: 30               # Timeout in seconds
  max_retries: 3                    # Retry failed requests
  delay_between_requests: 1         # Politeness delay (seconds)
  respect_robots_txt: true          # Honor robots.txt
```

### Incremental Crawling

```yaml
incremental:
  enabled: true                     # Enable incremental crawling
  change_detection: "hash"          # Method: "hash" or "last_modified"
  hash_algorithm: "sha256"          # Hash algorithm
  state_storage: "./data/crawler_state.db"  # State file location
  force_recrawl_after_days: 7       # Force re-crawl interval
```

### Content Extraction

```yaml
extraction:
  remove_elements:                  # CSS selectors to remove
    - "nav"
    - "header"
    - "footer"
    - ".sidebar"

  preserve_elements:                # Keep these elements
    - "pre"
    - "code"
    - "table"

  output_format: "markdown"         # Output format
  min_content_length: 100           # Skip short pages
```

### Metadata Extraction

```yaml
metadata:
  extract_meta_tags: true           # Extract HTML meta tags
  extract_from_url: true            # Parse URL structure

  custom_extractors:                # Custom CSS selectors
    component_name: "h1.component-title"
    category: ".breadcrumb li:last-child"
    version: ".version-selector .current"
```

## Output Format

Crawled pages are saved as Markdown files with YAML frontmatter:

```markdown
---
url: https://mui.com/material-ui/react-button/
crawled_at: '2024-01-15T10:30:00'
title: Button React component - Material UI
domain: mui.com
path: /material-ui/react-button/
component_name: Button
category: Inputs
---

# Button

Buttons allow users to take actions with a single tap...

## Basic Button

\`\`\`jsx
<Button variant="contained">Click me</Button>
\`\`\`

...
```

## How Incremental Crawling Works

1. **First Run**: Crawls all pages, computes content hash, saves state
2. **Subsequent Runs**:
   - Checks if URL was crawled before
   - Compares content hash
   - Only processes if:
     - URL is new
     - Content changed
     - Force re-crawl interval passed (default: 7 days)

This saves time and bandwidth by only processing what's actually changed!

## Troubleshooting

### Content is JavaScript-rendered

If the site uses heavy JavaScript and content is missing:

1. Enable browser mode in `config/crawler_config.yaml`:
   ```yaml
   crawler:
     use_browser: true
     browser_headless: true
   ```

2. Make sure Playwright is installed:
   ```bash
   playwright install
   ```

### Pages are being skipped

- Check `min_content_length` in config - it might be filtering out short pages
- Check `exclude_patterns` in your source configuration
- Look at crawler logs for specific skip reasons

### Rate limiting / Getting blocked

- Increase `delay_between_requests` in config
- Decrease `max_concurrent_requests`
- Check if site requires authentication

### State file corrupted

```bash
python src/ingestion/crawl_docs.py clear-state -y
```

Then re-run the crawler.

## Advanced Usage

### Programmatic Usage

```python
from src.ingestion.web_crawler import IncrementalDocCrawler

# Create crawler
crawler = IncrementalDocCrawler(config_path="config/crawler_config.yaml")

# Run crawler
crawler.run()
```

### Custom Metadata Extractors

Add custom CSS selectors to extract specific metadata:

```yaml
metadata:
  custom_extractors:
    # Extract component type from a specific class
    component_type: ".component-badge"

    # Extract framework from breadcrumbs
    framework: "nav.breadcrumb span:first-child"

    # Extract status (stable/beta/deprecated)
    status: ".status-indicator"
```

## Next Steps

After crawling documentation:

1. **Review Output**: Check `data/raw/crawled/` for quality
2. **Process Documents**: Move to Phase 2b - chunking and indexing
3. **Build Index**: Generate embeddings and store in vector database
4. **Query**: Use the RAG pipeline to search documentation

## Share Your URLs

Ready to add your documentation sources? Share the URLs you want to crawl, and we'll configure them!

Examples:
- Material UI: https://mui.com/material-ui/
- Ant Design: https://ant.design/components/
- Chakra UI: https://chakra-ui.com/docs/
- Carbon Design: https://carbondesignsystem.com/
- Your custom design system: https://...
