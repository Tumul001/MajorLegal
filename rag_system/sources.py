"""
Indian Legal Data Sources
Complete list of URLs for scraping Indian legal data
"""

class IndianLegalSources:
    """Centralized repository of Indian legal data sources"""
    
    # Best free sources for scraping
    RECOMMENDED_SOURCES = {
        'indiankanoon': {
            'url': 'https://indiankanoon.org',
            'search': 'https://indiankanoon.org/search/',
            'difficulty': 'Easy',
            'quality': 'Excellent',
            'api': 'No official API',
        },
        'sci_judgments': {
            'url': 'https://main.sci.gov.in/judgments',
            'difficulty': 'Medium',
            'quality': 'Official',
            'api': 'No',
        },
        'livelaw': {
            'url': 'https://www.livelaw.in/',
            'sc_url': 'https://www.livelaw.in/top-stories/supreme-court',
            'difficulty': 'Easy',
            'quality': 'Good with commentary',
            'api': 'No',
        }
    }
    
    # Supreme Court sources
    SUPREME_COURT = {
        'judgments': 'https://main.sci.gov.in/judgments',
        'daily_orders': 'https://main.sci.gov.in/daily-orders',
        'case_status': 'https://main.sci.gov.in/case-status',
        'efiling': 'https://efiling.sci.gov.in/',
    }
    
    # All High Courts
    HIGH_COURTS = {
        'Delhi': 'https://delhihighcourt.nic.in/judgments.asp',
        'Bombay': 'https://bombayhighcourt.nic.in/judgmentorder.php',
        'Madras': 'https://www.hcmadras.tn.nic.in/judis/',
        'Calcutta': 'https://calcuttahighcourt.gov.in/judgements-orders',
        'Karnataka': 'https://karnatakajudiciary.kar.nic.in/judgmentSearch.php',
        'Kerala': 'https://hckerala.gov.in/judgments',
        'Gujarat': 'https://gujarathighcourt.nic.in/Orders_Judgments',
        'Allahabad': 'https://allahabadhighcourt.in/event/judgement.html',
        'Rajasthan': 'https://hcraj.nic.in/hcjudgements/hcjindex.php',
        'Madhya Pradesh': 'https://mphc.gov.in/judgments.html',
        'Punjab & Haryana': 'https://highcourtchd.gov.in/?page_id=1267',
        'Patna': 'https://patnahighcourt.gov.in/judgements',
        'Chhattisgarh': 'https://highcourt.cg.gov.in/judgements/',
        'Telangana': 'https://hc.ts.nic.in/Judgements.aspx',
        'Jharkhand': 'https://jharkhandhighcourt.nic.in/judgments',
        'Uttarakhand': 'https://highcourtofuttarakhand.gov.in/judgement.html',
        'Himachal Pradesh': 'https://hphighcourt.nic.in/judgementslatest.asp',
        'Gauhati': 'https://ghconline.gov.in/Judgment/JudgmentSearch.aspx',
    }
    
    # National portals
    NATIONAL_PORTALS = {
        'njdg': 'https://njdg.ecourts.gov.in/njdgnew/',
        'ecourts': 'https://ecourts.gov.in/ecourts_home/',
        'ecourts_services': 'https://services.ecourts.gov.in/ecourtindia_v6/',
    }
    
    # Legislative sources
    LEGISLATIVE = {
        'india_code': 'https://www.indiacode.nic.in/',
        'acts_parliament': 'https://legislative.gov.in/acts-of-parliament-by-year',
        'law_commission': 'https://lawcommissionofindia.nic.in/reports.html',
    }
    
    # Open data repositories
    OPEN_DATA = {
        'datagov': 'https://data.gov.in/sector/justice-law-order',
        'opennyai': 'https://github.com/OpenNyAI/Opennyai',
        'kaggle': 'https://www.kaggle.com/search?q=indian+legal+data',
    }
    
    # News & commentary
    NEWS_SOURCES = {
        'livelaw': 'https://www.livelaw.in/',
        'barandbench': 'https://www.barandbench.com/',
        'legallyindia': 'https://www.legallyindia.com/',
        'scobserver': 'https://www.scobserver.in/',
    }
    
    # Paid services (for reference - require subscription)
    PAID_SERVICES = {
        'manupatra': 'https://www.manupatrafast.com/',
        'scconline': 'https://www.scconline.com/',
        'lexisnexis': 'https://www.lexisnexis.in/',
    }
    
    @staticmethod
    def get_scraping_priority():
        """
        Return sources in recommended scraping order
        
        Returns:
            List of (name, url, difficulty, notes) tuples
        """
        return [
            ('Indian Kanoon', 'https://indiankanoon.org', 'Easy', 'Best free source, well-structured'),
            ('LiveLaw', 'https://www.livelaw.in/', 'Easy', 'Recent cases with analysis'),
            ('Supreme Court', 'https://main.sci.gov.in/judgments', 'Medium', 'Official source'),
            ('Delhi HC', 'https://delhihighcourt.nic.in/judgments.asp', 'Medium', 'Regional coverage'),
            ('NJDG', 'https://njdg.ecourts.gov.in/njdgnew/', 'Hard', 'Comprehensive but complex'),
        ]
    
    @staticmethod
    def get_sample_searches():
        """Sample search queries for Indian Kanoon"""
        return [
            'arrest without warrant Section 41 CrPC',
            'Miranda rights interrogation',
            'search and seizure Fourth Amendment',
            'bail provisions CrPC',
            'criminal procedure code Section 154',
            'habeas corpus Article 32',
            'preventive detention Article 22',
            'fundamental rights Article 21',
        ]
    
    @staticmethod
    def get_robots_txt_url(source_name: str) -> str:
        """Get robots.txt URL for a source"""
        base_urls = {
            'indiankanoon': 'https://indiankanoon.org',
            'livelaw': 'https://www.livelaw.in',
            'sci': 'https://main.sci.gov.in',
        }
        
        base = base_urls.get(source_name, '')
        return f"{base}/robots.txt" if base else None