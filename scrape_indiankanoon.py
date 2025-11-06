"""
Scraper for Indian Kanoon - Best free source for Indian legal data
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from datetime import datetime
import re

class IndianKanoonScraper:
    def __init__(self):
        self.base_url = "https://indiankanoon.org"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.delay = 1.5  # Balanced delay to avoid 429 errors
        self.max_retries = 2  # Only retry twice, then skip to next case
        
    def search_cases(self, query: str, max_results: int = 10):
        """
        Search for cases on Indian Kanoon with pagination support
        
        Args:
            query: Search term (e.g., "arrest without warrant")
            max_results: Number of cases to retrieve (default: 10)
        
        Returns:
            List of case dictionaries
        """
        print(f"🔍 Searching for: {query}")
        
        cases = []
        page_num = 0
        consecutive_empty_pages = 0
        max_empty_pages = 3  # Stop after 3 consecutive empty pages
        
        while len(cases) < max_results and consecutive_empty_pages < max_empty_pages:
            # Build search URL with pagination
            search_url = f"{self.base_url}/search/?formInput={query.replace(' ', '%20')}&pagenum={page_num}"
            
            try:
                response = self._make_request_with_retry(search_url, retries=self.max_retries)
                if not response:
                    print(f"  ⚠️ Failed to fetch page {page_num} for: {query}")
                    consecutive_empty_pages += 1
                    page_num += 1
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find case links - they're in <div class="result"> with <a> tags containing /doc/
                case_results = soup.find_all('div', class_='result')
                
                if not case_results:
                    consecutive_empty_pages += 1
                    if page_num == 0:
                        print(f"  ⚠️ No results found for query: {query}")
                        return []
                    # No more results on this page, try next page
                    page_num += 1
                    continue
                
                # Reset empty page counter if we found results
                consecutive_empty_pages = 0
                cases_on_this_page = 0
                
                for result in case_results:
                    if len(cases) >= max_results:
                        break
                    
                    # Find the main case link within the result
                    result_links = result.find_all('a')
                    
                    for link in result_links:
                        href = link.get('href', '')
                        link_text = link.text.strip()
                        
                        # Filter: Skip statutory sections and acts, only get actual case judgments
                        if '/doc/' in href and link_text:
                            # Skip if it's a section or entire act
                            if 'Section ' in link_text or 'Entire Act' in link_text or 'Article ' in link_text:
                                continue
                            
                            # This looks like an actual case
                            cases_on_this_page += 1
                            print(f"  📄 Processing case {len(cases) + 1}/{max_results} (page {page_num})...")
                            
                            case_url = self.base_url + href if not href.startswith('http') else href
                            case_name = link_text
                            
                            # Get full case details
                            case_data = self.scrape_case_details(case_url, case_name)
                            if case_data:
                                cases.append(case_data)
                            
                            # Human-like delay with randomness to avoid detection
                            import random
                            delay = self.delay + random.uniform(0, 0.5)
                            time.sleep(delay)
                            break  # Only process first valid link per result
                
                # If no cases found on this page, it might be the end
                if cases_on_this_page == 0:
                    consecutive_empty_pages += 1
                
                # Move to next page
                page_num += 1
                
                # Small delay between pages (optimized)
                time.sleep(0.3)
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Error searching page {page_num}: {e}")
                consecutive_empty_pages += 1
                page_num += 1
                continue
            except Exception as e:
                print(f"  ❌ Unexpected error on page {page_num}: {e}")
                consecutive_empty_pages += 1
                page_num += 1
                continue
        
        print(f"  ✅ Successfully scraped {len(cases)} cases across {page_num} pages")
        return cases
    
    def _make_request_with_retry(self, url: str, retries: int = 2):
        """
        Make HTTP request with retry logic - retry twice then skip
        
        Args:
            url: URL to fetch
            retries: Number of retry attempts (default: 2)
            
        Returns:
            Response object or None if all retries fail
        """
        import random
        
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                # Handle 429 Too Many Requests - skip after 2 tries
                if e.response.status_code == 429:
                    if attempt < retries - 1:
                        wait_time = 3 + random.uniform(1, 2)  # 4-5 seconds
                        print(f"  ⚠️ Rate limited (429), waiting {wait_time:.1f}s (retry {attempt + 1}/{retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ⏭️ Rate limited, skipping case after {retries} attempts")
                        return None
                else:
                    print(f"  ❌ HTTP error {e.response.status_code}, skipping")
                    return None
            except (requests.exceptions.SSLError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt < retries - 1:
                    wait_time = 2  # Fixed 2 second wait
                    print(f"  ⚠️ Network error, retrying in {wait_time}s (attempt {attempt + 1}/{retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"  ⏭️ Network error after {retries} attempts, skipping: {type(e).__name__}")
                    return None
            except Exception as e:
                print(f"  ⏭️ Error, skipping: {e}")
                return None
        return None
    
    def scrape_case_details(self, url: str, case_name: str):
        """
        Scrape detailed information from a single case with retry logic
        
        Args:
            url: URL of the case page
            case_name: Name of the case
        
        Returns:
            Dictionary with case details or None if failed
        """
        try:
            response = self._make_request_with_retry(url, retries=self.max_retries)
            if not response:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract case metadata
            metadata = {}
            
            # Get citation
            citation_tag = soup.find('div', class_='doc_citations')
            if citation_tag:
                metadata['citation'] = citation_tag.text.strip()
            else:
                metadata['citation'] = 'N/A'
            
            # Get court name
            court_tag = soup.find('div', class_='doc_author')
            if court_tag:
                metadata['court'] = court_tag.text.strip()
            else:
                metadata['court'] = 'Supreme Court of India'
            
            # Get date
            date_tag = soup.find('div', class_='doc_date')
            if date_tag:
                metadata['date'] = date_tag.text.strip()
            else:
                metadata['date'] = '2020'
            
            # Get judges
            judges_tag = soup.find('div', class_='doc_bench')
            if judges_tag:
                metadata['judges'] = [j.strip() for j in judges_tag.text.split(',')]
            else:
                metadata['judges'] = []
            
            # Get full judgment text
            judgment_div = soup.find('div', class_='judgments')
            if judgment_div:
                # Remove script tags and extra whitespace
                for script in judgment_div.find_all('script'):
                    script.decompose()
                for style in judgment_div.find_all('style'):
                    style.decompose()
                text = judgment_div.get_text(separator='\n', strip=True)
            else:
                # Fallback: get all text
                text = soup.get_text(separator='\n', strip=True)
            
            # Extract acts mentioned
            acts_mentioned = self._extract_acts(text)
            
            # Extract sections
            sections = self._extract_sections(text)
            
            return {
                'case_name': case_name,
                'url': url,
                'citation': metadata['citation'],
                'court': metadata['court'],
                'date': metadata['date'],
                'judges': metadata['judges'],
                'acts_mentioned': acts_mentioned,
                'sections': sections,
                'text': text[:5000],  # First 5000 chars for preview
                'full_text': text,
                'scraped_at': datetime.now().isoformat(),
                'source': 'Indian Kanoon'
            }
            
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Network error scraping {case_name}: {e}")
            return None
        except Exception as e:
            print(f"    ⚠️ Error scraping {case_name}: {e}")
            return None
    
    def _extract_acts(self, text: str) -> list:
        """Extract mentioned acts from text"""
        acts = []
        
        # Common Indian acts patterns
        act_patterns = [
            r'Indian Penal Code',
            r'\bIPC\b',
            r'Code of Criminal Procedure',
            r'\bCrPC\b',
            r'Evidence Act',
            r'Constitution of India',
            r'Prevention of Corruption Act',
            r'NDPS Act',
            r'Arms Act',
            r'Explosive Substances Act',
        ]
        
        for pattern in act_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Clean up the match
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    acts.append(match.group())
        
        return list(set(acts))  # Remove duplicates
    
    def _extract_sections(self, text: str) -> list:
        """Extract section references from text"""
        sections = []
        
        # Match patterns like "Section 302 IPC", "Article 21", etc.
        patterns = [
            r'Section\s+\d+[A-Z]?(?:\s*\([a-z0-9]+\))?',
            r'Article\s+\d+[A-Z]?',
            r'Section\s+\d+\([a-z0-9]+\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend(matches)
        
        # Return unique sections, limited to top 15
        return list(set(sections))[:15]
    
    def save_cases(self, cases: list, filename: str = 'scraped_cases.json'):
        """
        Save cases to JSON file
        
        Args:
            cases: List of case dictionaries
            filename: Output filename (default: scraped_cases.json)
        
        Returns:
            Path to saved file
        """
        if not cases:
            print("⚠️ No cases to save")
            return None
        
        output_dir = Path('data/raw')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(cases)} cases to {output_file}")
        return output_file


def main():
    """Main scraping function"""
    print("="*70)
    print("INDIAN KANOON SCRAPER")
    print("="*70)
    
    scraper = IndianKanoonScraper()
    
    # Define search queries based on your case types
    queries = [
        "arrest without warrant Supreme Court",
        "criminal procedure fundamental rights",
        "search and seizure Article 21",
        "preventive detention Supreme Court",
        "bail provisions CrPC",
    ]
    
    print(f"\n📋 Will scrape {len(queries)} queries with 5 cases each")
    print(f"⏱️ Estimated time: ~{len(queries) * 5 * 3 / 60:.1f} minutes")
    
    proceed = input("\n✅ Proceed? (y/n): ")
    if proceed.lower() != 'y':
        print("❌ Scraping cancelled")
        return
    
    all_cases = []
    
    for idx, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {idx}/{len(queries)}: {query}")
        print('='*70)
        
        cases = scraper.search_cases(query, max_results=5)
        all_cases.extend(cases)
        
        # Be respectful with delays between queries
        if idx < len(queries):
            print(f"\n⏸️ Waiting 3 seconds before next query...")
            time.sleep(3)
    
    # Save all scraped cases
    if all_cases:
        scraper.save_cases(all_cases, 'indiankanoon_cases.json')
        
        print(f"\n{'='*70}")
        print(f"✅ SCRAPING COMPLETE!")
        print(f"{'='*70}")
        print(f"📊 Total cases scraped: {len(all_cases)}")
        print(f"📁 Saved to: data/raw/indiankanoon_cases.json")
        print(f"\n🚀 Next step: Run 'python build_from_scraped.py'")
    else:
        print("\n❌ No cases were scraped. Check your internet connection.")


if __name__ == "__main__":
    main()
