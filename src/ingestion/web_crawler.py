"""
Web crawler for design system documentation.
Supports incremental crawling with change detection.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import yaml
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
import html2text

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


class IncrementalDocCrawler:
    """Crawler for documentation sites with incremental update support."""

    def __init__(self, config_path: str = "config/crawler_config.yaml"):
        """Initialize the crawler with configuration."""
        self.config = self._load_config(config_path)
        self.state_db = self._load_state()
        self.html_to_markdown = html2text.HTML2Text()
        self._configure_html2text()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DesignSystemCrawler/1.0; +https://github.com/yourusername/rag-pipeline)'
        })
        self.visited_urls: Set[str] = set()

        # Initialize Playwright if browser mode is enabled
        self.playwright = None
        self.browser = None
        self.use_browser = self.config['crawler'].get('use_browser', False)

        if self.use_browser:
            if not PLAYWRIGHT_AVAILABLE:
                raise ImportError(
                    "Playwright is not installed. Install it with: "
                    "pip install playwright && playwright install"
                )
            logger.info("Initializing Playwright browser...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.config['crawler'].get('browser_headless', True)
            )
            logger.info("Browser initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load crawler configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _configure_html2text(self):
        """Configure HTML to Markdown converter."""
        self.html_to_markdown.ignore_links = False
        self.html_to_markdown.ignore_images = False
        self.html_to_markdown.ignore_emphasis = False
        self.html_to_markdown.body_width = 0  # Don't wrap lines
        self.html_to_markdown.protect_links = True
        self.html_to_markdown.mark_code = True

    def _load_state(self) -> Dict:
        """Load previous crawl state for incremental crawling."""
        state_path = Path(self.config['incremental']['state_storage'])

        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded crawl state with {len(state)} URLs")
            return state

        logger.info("No previous crawl state found - starting fresh")
        return {}

    def _save_state(self):
        """Save current crawl state."""
        state_path = Path(self.config['incremental']['state_storage'])
        state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_path, 'w') as f:
            json.dump(self.state_db, f, indent=2)

        logger.info(f"Saved crawl state with {len(self.state_db)} URLs")

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content for change detection."""
        algorithm = self.config['incremental']['hash_algorithm']
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode('utf-8'))
        return hash_obj.hexdigest()

    def _should_process_url(self, url: str, content: str) -> bool:
        """Determine if URL should be processed based on incremental settings."""
        if not self.config['incremental']['enabled']:
            return True

        # Check if URL was crawled before
        if url not in self.state_db:
            logger.debug(f"New URL: {url}")
            return True

        previous_state = self.state_db[url]

        # Check if forced re-crawl interval has passed
        last_crawled = datetime.fromisoformat(previous_state['last_crawled'])
        force_recrawl_days = self.config['incremental']['force_recrawl_after_days']
        if datetime.now() - last_crawled > timedelta(days=force_recrawl_days):
            logger.debug(f"Force re-crawl (age): {url}")
            return True

        # Check if content has changed
        content_hash = self._compute_content_hash(content)
        if content_hash != previous_state.get('content_hash'):
            logger.debug(f"Content changed: {url}")
            return True

        logger.debug(f"Skipping unchanged: {url}")
        return False

    def _extract_main_content(self, html: str, url: str) -> str:
        """Extract main content from HTML using readability."""
        try:
            doc = ReadabilityDocument(html)
            main_content = doc.summary()
            return main_content
        except Exception as e:
            logger.warning(f"Readability extraction failed for {url}: {e}")
            return html

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove navigation, headers, footers, etc."""
        # Make a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), 'html.parser')

        # Remove elements specified in config
        for selector in self.config['extraction']['remove_elements']:
            for element in soup_copy.select(selector):
                element.decompose()

        return soup_copy

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from HTML and URL."""
        metadata = {
            'url': url,
            'crawled_at': datetime.now().isoformat(),
        }

        # Extract from HTML meta tags
        if self.config['metadata']['extract_meta_tags']:
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()

            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    metadata[name] = content

        # Extract from URL structure
        if self.config['metadata']['extract_from_url']:
            parsed_url = urlparse(url)
            metadata['domain'] = parsed_url.netloc
            metadata['path'] = parsed_url.path

            # Try to extract component name from path
            path_parts = [p for p in parsed_url.path.split('/') if p]
            if path_parts:
                metadata['path_segment'] = path_parts[-1]

        # Custom extractors from config
        for key, selector in self.config['metadata'].get('custom_extractors', {}).items():
            element = soup.select_one(selector)
            if element:
                metadata[key] = element.get_text().strip()

        return metadata

    def _is_allowed_url(self, url: str, source_config: Dict) -> bool:
        """Check if URL is allowed to be crawled."""
        parsed = urlparse(url)

        # Check domain
        allowed_domains = source_config.get('allowed_domains', [])
        if allowed_domains and parsed.netloc not in allowed_domains:
            return False

        # Check URL patterns
        url_patterns = source_config.get('url_patterns', [])
        if url_patterns:
            import fnmatch
            if not any(fnmatch.fnmatch(url, pattern) for pattern in url_patterns):
                return False

        # Check exclude patterns
        exclude_patterns = source_config.get('exclude_patterns', [])
        if exclude_patterns:
            import fnmatch
            if any(fnmatch.fnmatch(url, pattern) for pattern in exclude_patterns):
                return False

        return True

    def _extract_links(self, soup: BeautifulSoup, base_url: str, source_config: Dict) -> List[str]:
        """Extract and filter links from page."""
        links = []

        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']

            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)

            # Remove fragments
            absolute_url = absolute_url.split('#')[0]

            # Check if allowed
            if self._is_allowed_url(absolute_url, source_config):
                links.append(absolute_url)

        return list(set(links))  # Remove duplicates

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content."""
        if self.use_browser:
            return self._fetch_page_with_browser(url)
        else:
            return self._fetch_page_with_requests(url)

    def _fetch_page_with_requests(self, url: str) -> Optional[str]:
        """Fetch page content using requests library."""
        try:
            response = self.session.get(
                url,
                timeout=self.config['crawler']['request_timeout']
            )
            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _fetch_page_with_browser(self, url: str) -> Optional[str]:
        """Fetch page content using Playwright browser."""
        try:
            page = self.browser.new_page()
            page.goto(url, timeout=self.config['crawler']['request_timeout'] * 1000)
            # Wait for network to be idle to ensure JavaScript has loaded
            page.wait_for_load_state('networkidle', timeout=30000)
            content = page.content()
            page.close()
            return content

        except Exception as e:
            logger.error(f"Failed to fetch {url} with browser: {e}")
            return None

    def _crawl_page(self, url: str, source_config: Dict, depth: int):
        """Crawl a single page."""
        # Check if already visited
        if url in self.visited_urls:
            return

        # Check depth limit
        max_depth = source_config.get('max_depth', 3)
        if depth > max_depth:
            logger.debug(f"Max depth reached for: {url}")
            return

        # Check max requests
        if len(self.visited_urls) >= self.config['crawler']['max_requests_per_crawl']:
            logger.warning("Max requests limit reached")
            return

        logger.info(f"[Depth {depth}] Crawling: {url}")
        self.visited_urls.add(url)

        # Fetch page
        html = self._fetch_page(url)
        if not html:
            return

        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Clean HTML
        cleaned_soup = self._clean_html(soup)

        # Extract main content
        main_content_html = self._extract_main_content(str(cleaned_soup), url)

        # Convert to markdown
        markdown_content = self.html_to_markdown.handle(main_content_html)

        # Check content length
        min_length = self.config['extraction']['min_content_length']
        if len(markdown_content.strip()) < min_length:
            logger.warning(f"Content too short ({len(markdown_content)} chars), skipping: {url}")
            # Still extract links for deeper crawling
            links = self._extract_links(soup, url, source_config)
            for link in links:
                time.sleep(self.config['crawler']['delay_between_requests'])
                self._crawl_page(link, source_config, depth + 1)
            return

        # Check if should process (incremental crawling)
        if not self._should_process_url(url, markdown_content):
            # Still extract links for deeper crawling
            links = self._extract_links(soup, url, source_config)
            for link in links:
                time.sleep(self.config['crawler']['delay_between_requests'])
                self._crawl_page(link, source_config, depth + 1)
            return

        # Extract metadata
        metadata = self._extract_metadata(soup, url)

        # Save content
        output_dir = Path(self.config['output']['raw_output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        output_file = output_dir / f"{url_hash}.md"

        with open(output_file, 'w', encoding='utf-8') as f:
            # Write metadata as frontmatter
            f.write("---\n")
            f.write(yaml.dump(metadata, default_flow_style=False))
            f.write("---\n\n")
            f.write(markdown_content)

        logger.info(f"✓ Saved: {output_file}")

        # Update state
        self.state_db[url] = {
            'content_hash': self._compute_content_hash(markdown_content),
            'last_crawled': datetime.now().isoformat(),
            'output_file': str(output_file),
            'metadata': metadata,
        }

        # Extract and crawl links
        links = self._extract_links(soup, url, source_config)
        logger.debug(f"Found {len(links)} links on {url}")

        for link in links:
            # Delay between requests (be polite)
            time.sleep(self.config['crawler']['delay_between_requests'])
            self._crawl_page(link, source_config, depth + 1)

    def crawl_source(self, source_config: Dict):
        """Crawl a single documentation source."""
        logger.info(f"Starting crawl for: {source_config['name']}")
        logger.info(f"Starting URL: {source_config['start_url']}")

        self.visited_urls = set()  # Reset for each source
        self._crawl_page(source_config['start_url'], source_config, depth=0)

        logger.info(f"Completed crawl for: {source_config['name']}")
        logger.info(f"Total pages crawled: {len(self.visited_urls)}")

    def crawl_all_sources(self):
        """Crawl all configured documentation sources."""
        sources = self.config.get('sources', [])

        if not sources:
            logger.warning("No sources configured in crawler_config.yaml")
            logger.info("Please add documentation sources to config/crawler_config.yaml")
            return

        for source in sources:
            try:
                self.crawl_source(source)
            except Exception as e:
                logger.error(f"Error crawling {source['name']}: {e}", exc_info=True)

        # Save final state
        self._save_state()
        logger.info("All crawls completed")

    def cleanup(self):
        """Clean up resources."""
        if self.browser:
            logger.info("Closing browser...")
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def run(self):
        """Run the crawler."""
        try:
            self.crawl_all_sources()
        finally:
            self.cleanup()


def main():
    """Main entry point for running the crawler."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    crawler = IncrementalDocCrawler()
    crawler.run()


if __name__ == "__main__":
    main()
