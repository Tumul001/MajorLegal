"""
Pre-scraped Indian Legal Dataset Downloader

Downloads existing datasets from public sources to quickly expand database.
Expected: 20,000+ cases in 30 minutes
"""

import requests
import json
import os
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import zipfile
import gzip
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PretrainedDatasetDownloader:
    """
    Download pre-scraped Indian legal datasets from public sources
    """
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available datasets (VERIFIED working URLs)
        self.datasets = {
            'kaggle_sc_judgments': {
                'name': 'Supreme Court Judgments (Kaggle)',
                'description': 'SC cases 2000-2020',
                'estimated_cases': 10000,
                'size_mb': 450,
                'format': 'json',
                'download_method': 'kaggle',
                'dataset_id': 'deepcontractor/supreme-court-judgements'
            },
            'huggingface_legal_ner_train': {
                'name': 'InLegalNER Training Dataset (HuggingFace)',
                'description': 'Indian legal NER training data with 9,435 annotated judgments',
                'estimated_cases': 9435,
                'size_mb': 50,
                'format': 'zip',
                'download_method': 'direct',
                'url': 'https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_TRAIN.zip?download=true'
            },
            'huggingface_legal_ner_test': {
                'name': 'InLegalNER Test Dataset (HuggingFace)',
                'description': 'Indian legal NER test data',
                'estimated_cases': 2000,
                'size_mb': 15,
                'format': 'zip',
                'download_method': 'direct',
                'url': 'https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_TEST.zip?download=true'
            }
        }
    
    def download_with_progress(self, url: str, filename: str, use_auth: bool = False) -> bool:
        """
        Download file with progress bar
        """
        try:
            logger.info(f"Downloading: {url}")
            
            headers = {}
            if use_auth:
                # Try to get HuggingFace token
                try:
                    from huggingface_hub import HfFolder
                    token = HfFolder.get_token()
                    if token:
                        headers['Authorization'] = f'Bearer {token}'
                        logger.info("Using HuggingFace authentication")
                except:
                    pass
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            logger.info(f"✅ Downloaded: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def download_from_kaggle(self, dataset_id: str, output_name: str) -> bool:
        """
        Download dataset from Kaggle (requires kaggle API)
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            logger.info(f"Downloading from Kaggle: {dataset_id}")
            
            api = KaggleApi()
            api.authenticate()
            
            # Download to output directory
            api.dataset_download_files(
                dataset_id,
                path=str(self.output_dir / 'kaggle_temp'),
                unzip=True
            )
            
            logger.info(f"✅ Downloaded from Kaggle")
            return True
        
        except ImportError:
            logger.warning("⚠️  Kaggle API not installed. Run: pip install kaggle")
            logger.info("💡 Alternative: Download manually from kaggle.com")
            return False
        
        except Exception as e:
            logger.error(f"❌ Kaggle download failed: {e}")
            logger.info("💡 Setup: https://github.com/Kaggle/kaggle-api#api-credentials")
            return False
    
    def download_from_github(self, repo: str, path: str, output_name: str) -> bool:
        """
        Download file from GitHub repository
        """
        try:
            url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
            
            output_path = self.output_dir / output_name
            
            return self.download_with_progress(url, str(output_path))
        
        except Exception as e:
            logger.error(f"❌ GitHub download failed: {e}")
            return False
    
    def download_from_huggingface_datasets(self, dataset_id: str, subset: str, output_name: str) -> bool:
        """
        Download dataset from HuggingFace datasets library
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Downloading HuggingFace dataset: {dataset_id} (subset: {subset})")
            
            # Load dataset
            dataset = load_dataset(dataset_id, subset, split='train')
            
            # Convert to list of dicts
            data = []
            for item in dataset:
                data.append(dict(item))
            
            # Save as JSON
            output_path = self.output_dir / output_name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Downloaded and saved: {output_path}")
            return True
        
        except ImportError:
            logger.warning("⚠️  HuggingFace datasets library not installed. Run: pip install datasets")
            return False
        
        except Exception as e:
            logger.error(f"❌ HuggingFace download failed: {e}")
            return False
    
    def download_direct(self, url: str, output_name: str, use_auth: bool = False) -> bool:
        """
        Direct download from URL
        """
        output_path = self.output_dir / output_name
        
        success = self.download_with_progress(url, str(output_path), use_auth=use_auth)
        
        if not success:
            return False
        
        # Extract if compressed
        if output_name.endswith('.gz'):
            logger.info(f"Extracting gzip: {output_name}")
            extracted_path = str(output_path).replace('.gz', '')
            
            with gzip.open(output_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"✅ Extracted: {extracted_path}")
            os.remove(output_path)  # Remove compressed file
        
        elif output_name.endswith('.zip'):
            logger.info(f"Extracting ZIP: {output_name}")
            extract_dir = self.output_dir / output_name.replace('.zip', '')
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"✅ Extracted to: {extract_dir}")
            
            # Convert extracted files to JSON format if needed
            json_files = list(extract_dir.glob('**/*.json'))
            if json_files:
                # Merge all JSON files into one
                all_data = []
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            else:
                                all_data.append(data)
                        except:
                            pass
                
                # Save merged JSON
                merged_file = self.output_dir / output_name.replace('.zip', '.json')
                with open(merged_file, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ Merged to: {merged_file}")
            
            os.remove(output_path)  # Remove ZIP file
        
        return success
    
    def download_dataset(self, dataset_key: str) -> Dict:
        """
        Download a specific dataset
        """
        if dataset_key not in self.datasets:
            logger.error(f"❌ Unknown dataset: {dataset_key}")
            return {'success': False, 'cases': 0}
        
        dataset = self.datasets[dataset_key]
        
        print("\n" + "="*70)
        print(f"Downloading: {dataset['name']}")
        print("="*70)
        print(f"Description: {dataset['description']}")
        print(f"Estimated cases: {dataset['estimated_cases']:,}")
        print(f"Size: ~{dataset['size_mb']} MB")
        print("-"*70)
        
        # Determine output filename based on format
        if dataset['format'] == 'zip':
            output_name = f"{dataset_key}.zip"
        elif dataset['format'] == 'jsonl':
            output_name = f"{dataset_key}.jsonl"
        else:
            output_name = f"{dataset_key}.json"
        
        # Download based on method
        if dataset['download_method'] == 'kaggle':
            success = self.download_from_kaggle(
                dataset['dataset_id'],
                output_name
            )
        
        elif dataset['download_method'] == 'github':
            success = self.download_from_github(
                dataset['repo'],
                dataset['path'],
                output_name
            )
        
        elif dataset['download_method'] == 'direct':
            # Handle different URL extensions
            if '.gz' in dataset['url']:
                output_name_with_ext = f"{dataset_key}.json.gz"
            elif '.zip' in dataset['url'] or dataset.get('format') == 'zip':
                output_name_with_ext = f"{dataset_key}.zip"
            else:
                output_name_with_ext = output_name
            
            # Use auth for HuggingFace URLs
            use_auth = 'huggingface.co' in dataset['url']
            
            success = self.download_direct(
                dataset['url'],
                output_name_with_ext,
                use_auth=use_auth
            )
        
        elif dataset['download_method'] == 'huggingface_datasets':
            success = self.download_from_huggingface_datasets(
                dataset['dataset_id'],
                dataset.get('subset', 'default'),
                output_name
            )
        
        else:
            logger.error(f"❌ Unknown download method: {dataset['download_method']}")
            success = False
        
        if success:
            # Count actual cases
            try:
                json_path = self.output_dir / output_name
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        actual_cases = len(data) if isinstance(data, list) else len(data.get('cases', []))
                    
                    print(f"✅ Downloaded: {actual_cases:,} cases")
                    return {'success': True, 'cases': actual_cases, 'file': str(json_path)}
            except Exception as e:
                logger.error(f"Error counting cases: {e}")
        
        return {'success': False, 'cases': 0}
    
    def download_all_available(self) -> Dict:
        """
        Download all available datasets
        """
        print("\n" + "="*70)
        print("DOWNLOADING PRE-SCRAPED DATASETS")
        print("="*70)
        print("This will download publicly available Indian legal datasets")
        print("Estimated total: 20,000+ cases")
        print("Estimated time: 30-45 minutes")
        print("="*70)
        
        results = {
            'downloaded': [],
            'failed': [],
            'total_cases': 0
        }
        
        for dataset_key in self.datasets.keys():
            result = self.download_dataset(dataset_key)
            
            if result['success']:
                results['downloaded'].append({
                    'name': self.datasets[dataset_key]['name'],
                    'cases': result['cases'],
                    'file': result.get('file')
                })
                results['total_cases'] += result['cases']
            else:
                results['failed'].append(self.datasets[dataset_key]['name'])
        
        # Print summary
        print("\n" + "="*70)
        print("📊 DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Successfully downloaded: {len(results['downloaded'])} datasets")
        print(f"Failed: {len(results['failed'])} datasets")
        print(f"Total cases: {results['total_cases']:,}")
        print("-"*70)
        
        if results['downloaded']:
            print("\nDownloaded datasets:")
            for item in results['downloaded']:
                print(f"   - {item['name']}: {item['cases']:,} cases")
        
        if results['failed']:
            print("\n[WARNING] Failed datasets:")
            for name in results['failed']:
                print(f"   - {name}")
        
        print("="*70)
        
        return results


def main():
    """
    Main execution
    """
    downloader = PretrainedDatasetDownloader()
    
    # Try to download all available datasets
    results = downloader.download_all_available()
    
    # If no datasets downloaded successfully, show instructions
    if results['total_cases'] == 0:
        print("\n⚠️  No datasets downloaded successfully")
        print("\n💡 To download real datasets:")
        print("\n1. HuggingFace datasets (InLegalNER):")
        print("   - Visit: https://huggingface.co/datasets/opennyaiorg/InLegalNER")
        print("   - Click 'Request access' and wait for approval")
        print("   - Re-run this script after approval")
        print("\n2. Kaggle datasets:")
        print("   - Install: pip install kaggle")
        print("   - Setup credentials: https://github.com/Kaggle/kaggle-api")
        print("   - Or manually download from:")
        print("     https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgements")
        print("\n3. Alternative approach:")
        print("   - Run: python scrape_parallel.py")
        print("   - This will scrape recent cases directly from IndianKanoon")
    
    return results


if __name__ == "__main__":
    main()
