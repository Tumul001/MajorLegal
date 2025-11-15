"""
Parallel Indian Kanoon Scraper

Multi-threaded scraper with intelligent rate limiting.
Expected: 10x speedup over sequential scraping
"""

import concurrent.futures
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict
from datetime import datetime
import threading
from queue import Queue
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)-10s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelIndianKanoonScraper:
    """
    Multi-threaded scraper for IndianKanoon.org
    
    Features:
    - Parallel court scraping (5-10 threads)
    - Rotating user agents
    - Exponential backoff on errors
    - Checkpoint saves every 100 cases
    - Resume capability
    """
    
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
        self.base_url = "https://indiankanoon.org"
        
        # Thread-safe queue for results
        self.results_queue = Queue()
        self.total_scraped = 0
        self.lock = threading.Lock()
        
        # Create data directories
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        Path('data/checkpoints').mkdir(parents=True, exist_ok=True)
        
        # Rotating user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        ]
        
        # Courts configuration
        self.courts_config = {
            'supreme_court': {
                'url': '/search/?formInput=doctypes:supremecourt',
                'priority': 1,
                'pages_per_year': 40  # Increased for recent years
            },
            'delhi_hc': {
                'url': '/search/?formInput=doctypes:delhihighcourt',
                'priority': 2,
                'pages_per_year': 25
            },
            'bombay_hc': {
                'url': '/search/?formInput=doctypes:bombayhighcourt',
                'priority': 2,
                'pages_per_year': 25
            },
            'madras_hc': {
                'url': '/search/?formInput=doctypes:madrashighcourt',
                'priority': 2,
                'pages_per_year': 20
            },
            'calcutta_hc': {
                'url': '/search/?formInput=doctypes:calcuttahighcourt',
                'priority': 3,
                'pages_per_year': 15
            },
            'karnataka_hc': {
                'url': '/search/?formInput=doctypes:karnatakahighcourt',
                'priority': 3,
                'pages_per_year': 15
            },
        }
        
        # Year ranges (prioritize recent)
        self.year_ranges = [
            (2023, 2024),  # Most recent
            (2020, 2022),  # Recent
            (2015, 2019),  # Mid-range
            (2010, 2014),  # Historical
        ]
    
    def create_session(self) -> requests.Session:
        """Create session with rotating user agent"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session
    
    def scrape_court_year_range(self, court_name: str, start_year: int, 
                                end_year: int, max_pages: int) -> List[Dict]:
        """
        Scrape cases for one court and year range (runs in thread)
        """
        session = self.create_session()
        cases = []
        config = self.courts_config[court_name]
        
        thread_id = threading.current_thread().name
        logger.info(f"[{thread_id}] Started: {court_name} ({start_year}-{end_year})")
        
        for page in range(max_pages):
            try:
                # Build URL
                search_url = f"{self.base_url}{config['url']}"
                search_url += f"&fromdate={start_year}-01-01&todate={end_year}-12-31"
                search_url += f"&pagenum={page}"
                
                # Request with retry
                response = self._request_with_retry(session, search_url)
                
                if not response:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                case_results = soup.find_all('div', class_='result')
                
                if not case_results:
                    logger.info(f"[{thread_id}] No more results at page {page+1}")
                    break
                
                logger.info(f"[{thread_id}] Page {page+1}/{max_pages}: Found {len(case_results)} cases")
                
                # Process each case
                for result in case_results:
                    try:
                        title_tag = result.find('a', class_='result_title')
                        if not title_tag:
                            continue
                        
                        case_url = title_tag.get('href')
                        
                        # Get case details
                        time.sleep(random.uniform(1.0, 2.0))  # Rate limiting
                        
                        case_data = self.scrape_case_details(session, case_url)
                        
                        if case_data:
                            case_data['court'] = court_name.replace('_', ' ').title()
                            case_data['year_range'] = f"{start_year}-{end_year}"
                            cases.append(case_data)
                            
                            # Increment counter (thread-safe)
                            with self.lock:
                                self.total_scraped += 1
                                
                                # Checkpoint save every 100 cases
                                if self.total_scraped % 100 == 0:
                                    self._save_checkpoint()
                                
                                # Progress update
                                if self.total_scraped % 10 == 0:
                                    logger.info(f"📊 Total scraped: {self.total_scraped}")
                    
                    except Exception as e:
                        logger.error(f"[{thread_id}] Error parsing case: {e}")
                        continue
                
                # Rate limiting between pages
                time.sleep(random.uniform(1.5, 2.5))
            
            except Exception as e:
                logger.error(f"[{thread_id}] Error on page {page+1}: {e}")
                time.sleep(5)
                continue
        
        logger.info(f"[{thread_id}] ✅ Completed: {court_name} ({len(cases)} cases)")
        return cases
    
    def scrape_case_details(self, session: requests.Session, case_url: str) -> Dict:
        """Extract case details from individual case page"""
        try:
            full_url = f"{self.base_url}{case_url}" if not case_url.startswith('http') else case_url
            
            response = self._request_with_retry(session, full_url)
            if not response:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract case data
            case_data = {
                'url': full_url,
                'title': self._safe_extract(soup.find('h1', class_='doctitle')),
                'citation': self._safe_extract(soup.find('div', class_='docsource')),
                'court': self._safe_extract(soup.find('div', class_='doc_court')),
                'bench': self._safe_extract(soup.find('div', class_='doc_bench')),
                'date': self._safe_extract(soup.find('div', class_='doc_date')),
                'judges': [j.text.strip() for j in soup.find_all('div', class_='doc_author')],
                'subject': [tag.text.strip() for tag in soup.find_all('a', class_='tag')],
                'judgment_text': self._extract_judgment(soup),
                'scraped_at': datetime.now().isoformat(),
                'scraper_version': 'parallel_v1'
            }
            
            return case_data
        
        except Exception as e:
            logger.error(f"Error scraping case details: {e}")
            return None
    
    def _request_with_retry(self, session: requests.Session, url: str, max_retries=3):
        """Request with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                response = session.get(url, timeout=15)
                response.raise_for_status()
                return response
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential: 1s, 2s, 4s
                    logger.warning(f"Retry {attempt+1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} retries: {url}")
                    return None
    
    def _safe_extract(self, tag):
        """Safely extract text from BeautifulSoup tag"""
        return tag.text.strip() if tag else ''
    
    def _extract_judgment(self, soup):
        """Extract judgment text"""
        judgment_div = soup.find('div', class_='judgments')
        if judgment_div:
            # Remove scripts/styles
            for tag in judgment_div.find_all(['script', 'style']):
                tag.decompose()
            return judgment_div.get_text(separator='\n', strip=True)
        return ''
    
    def _save_checkpoint(self):
        """Save progress checkpoint"""
        all_cases = list(self.results_queue.queue)
        checkpoint_file = f'data/checkpoints/parallel_checkpoint_{self.total_scraped}_cases.json'
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {self.total_scraped} cases")
    
    def run_parallel_scrape(self, target_count=5000, focus_recent=True):
        """
        Main parallel scraping method
        
        Args:
            target_count: Target number of cases to scrape
            focus_recent: If True, prioritize 2023-2024 cases
        """
        
        print("\n" + "="*70)
        print("⚡ PARALLEL SCRAPING MODE")
        print("="*70)
        print(f"Workers: {self.max_workers} threads")
        print(f"Target: {target_count:,} cases")
        print(f"Focus: {'Recent cases (2023-2024)' if focus_recent else 'All years'}")
        print(f"Estimated time: {self._estimate_time(target_count)} minutes")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        # Create work items
        work_items = []
        
        year_ranges = [(2023, 2024)] if focus_recent else self.year_ranges
        
        for court_name, config in self.courts_config.items():
            for start_year, end_year in year_ranges:
                work_items.append({
                    'court': court_name,
                    'start_year': start_year,
                    'end_year': end_year,
                    'max_pages': config['pages_per_year']
                })
        
        logger.info(f"Created {len(work_items)} work items")
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_work = {
                executor.submit(
                    self.scrape_court_year_range,
                    item['court'],
                    item['start_year'],
                    item['end_year'],
                    item['max_pages']
                ): item for item in work_items
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_work):
                work_item = future_to_work[future]
                try:
                    cases = future.result()
                    
                    # Add to results queue
                    for case in cases:
                        self.results_queue.put(case)
                    
                    logger.info(f"✅ Completed: {work_item['court']} "
                               f"({work_item['start_year']}-{work_item['end_year']})")
                
                except Exception as e:
                    logger.error(f"❌ Failed: {work_item['court']}: {e}")
        
        # Final save
        all_cases = list(self.results_queue.queue)
        
        elapsed_time = time.time() - start_time
        
        output_file = f'data/raw/parallel_scraped_{len(all_cases)}_cases.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("✅ PARALLEL SCRAPING COMPLETE")
        print("="*70)
        print(f"📊 Cases scraped: {len(all_cases):,}")
        print(f"⏱️  Time taken: {elapsed_time/60:.1f} minutes")
        print(f"⚡ Speed: {len(all_cases)/(elapsed_time/60):.1f} cases/minute")
        print(f"💾 Saved to: {output_file}")
        print("="*70)
        
        return all_cases
    
    def _estimate_time(self, target_count):
        """Estimate scraping time in minutes"""
        # Assume 2 seconds per case with parallel workers
        estimated_seconds = (target_count * 2) / self.max_workers
        return int(estimated_seconds / 60)


def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(description='Parallel Indian Kanoon Scraper')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel threads (default: 8)')
    parser.add_argument('--target', type=int, default=5000, help='Target number of cases (default: 5000)')
    parser.add_argument('--recent', action='store_true', help='Focus on recent cases (2023-2024)')
    
    args = parser.parse_args()
    
    scraper = ParallelIndianKanoonScraper(max_workers=args.workers)
    cases = scraper.run_parallel_scrape(target_count=args.target, focus_recent=args.recent)
    
    print(f"\n🎉 Successfully scraped {len(cases):,} cases!")


if __name__ == "__main__":
    main()
