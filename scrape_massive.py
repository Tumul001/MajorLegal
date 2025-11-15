"""
MASSIVE Bulk Scraper for Indian Kanoon - Target: 50,000+ Cases
Comprehensive coverage of Indian legal system with 200+ queries
"""
import time
import json
from pathlib import Path
from scrape_indiankanoon import IndianKanoonScraper

def get_massive_queries():
    """
    Returns 200+ diverse legal queries covering ALL major Indian law areas
    Target: 200 queries × 250 cases per query = 50,000+ cases
    """
    return [
        # === CRIMINAL LAW & PROCEDURE (40 queries) ===
        
        # CrPC Sections
        "Section 154 FIR registration",
        "Section 156 police investigation",
        "Section 161 statement to police",
        "Section 164 confessional statement magistrate",
        # "Section 167 police custody remand",
        "Section 170 police report cognizable offence",
        "Section 173 investigation completion report",
        "Section 197 prosecution sanction public servant",
        "Section 207 supply documents accused",
        "Section 227 discharge accused insufficient evidence",
        "Section 228 framing charges trial",
        "Section 239 discharge insufficient grounds",
        "Section 313 statement accused trial",
        "Section 319 additional accused summoning",
        "Section 320 compounding offences",
        "Section 377 appeal criminal conviction",
        "Section 389 suspension sentence pending appeal",
        "Section 397 calling record lower court",
        "Section 401 High Court revisional jurisdiction",
        "Section 406 transfer investigation",
        "Section 432 remission sentence",
        "Section 437 bail bailable offence",
        "Section 438 anticipatory bail",
        "Section 439 bail High Court Sessions",
        "Section 482 quashing proceedings inherent powers",
        
        # IPC Major Sections
        "Section 120B criminal conspiracy IPC",
        "Section 302 murder life imprisonment",
        "Section 304 culpable homicide not murder",
        "Section 307 attempt to murder",
        "Section 376 rape sexual assault",
        "Section 377 unnatural offences",
        "Section 420 cheating dishonestly",
        "Section 468 forgery documents",
        "Section 471 using forged document",
        "Section 498A dowry harassment",
        "Section 506 criminal intimidation",
        
        # Bail & Custody
        "anticipatory bail grounds rejection",
        "bail cancellation Supreme Court",
        "default bail 60 days police custody",
        
        # === CONSTITUTION OF INDIA (35 queries) ===
        
        # Fundamental Rights
        "Article 14 equality before law",
        "Article 15 discrimination prohibition",
        "Article 19 freedom speech expression",
        "Article 20 double jeopardy protection",
        "Article 21 right to life personal liberty",
        "Article 21A right to education",
        "Article 22 protection arrest detention",
        "Article 23 trafficking prohibition forced labour",
        "Article 25 freedom religion",
        "Article 29 cultural educational rights minorities",
        "Article 32 constitutional remedies writs",
        
        # Directive Principles
        "Article 39A free legal aid",
        "Article 47 public health nutrition",
        
        # Writs & Remedies
        "habeas corpus Article 226 High Court",
        "mandamus writ public duty",
        "certiorari writ judicial review",
        "prohibition writ jurisdiction",
        "quo warranto writ public office",
        "PIL public interest litigation locus standi",
        
        # Constitutional Amendments
        "Article 368 constitutional amendment power",
        "basic structure doctrine Kesavananda Bharati",
        "42nd Amendment emergency constitutional",
        "44th Amendment fundamental rights restoration",
        "73rd Amendment panchayati raj",
        "74th Amendment municipalities urban governance",
        "86th Amendment right to education",
        "97th Amendment cooperative societies",
        "101st Amendment GST constitutional",
        "103rd Amendment economically weaker sections reservation",
        
        # Federal Structure
        "Article 356 President's Rule state",
        "Article 360 financial emergency",
        "Article 370 Jammu Kashmir special status",
        
        # === EVIDENCE ACT (20 queries) ===
        
        "Section 3 interpretation evidence facts",
        "Section 8 motive preparation conduct",
        "Section 24 confession caused inducement",
        "Section 25 confession to police officer",
        "Section 26 confession police custody",
        "Section 27 discovery fact confession",
        "Section 32 dying declaration deceased person",
        "Section 45 expert opinion evidence",
        "Section 60 oral evidence direct",
        "Section 65 secondary evidence documents",
        "Section 113A presumption dowry death",
        "Section 113B presumption dowry harassment",
        "Section 114 court presumption fact",
        "Section 118 witness competency",
        "Section 134 witness number sufficient",
        "Section 145 witness cross examination",
        "Section 146 previous statements impeach credit",
        "Section 154 refreshing memory witness",
        "Section 165 judge questions witness",
        "hostile witness cross examination prosecution",
        
        # === CIVIL PROCEDURE CODE (25 queries) ===
        
        "Order 1 Rule 10 impleading parties",
        "Order 6 Rule 17 amendment pleadings",
        "Order 7 Rule 11 rejection plaint",
        "Order 8 Rule 1 written statement filing",
        "Order 8 Rule 5 specific denial pleadings",
        "Order 9 ex parte decree setting aside",
        "Order 12 admission facts documents",
        "Order 14 settlement issues trial",
        "Order 18 Rule 2 recording evidence",
        "Order 20 Rule 12 judgment pronouncement",
        "Order 21 execution decree attachment",
        "Order 39 Rule 1 temporary injunction",
        "Order 41 appeals from original decree",
        "Section 9 CPC civil court jurisdiction",
        "Section 10 res judicata bar subsequent suit",
        "Section 11 res judicata explanation",
        "Section 34 CPC jurisdiction concurrent",
        "Section 47 questions execution decree",
        "Section 80 notice government suit",
        "Section 96 appeal decree",
        "Section 100 second appeal High Court",
        "Section 115 revision civil court",
        "Section 151 inherent powers civil court",
        "limitation period suit civil",
        "restitution decree reversed appeal",
        
        # === SPECIAL ACTS (40 queries) ===
        
        # NDPS Act
        "NDPS Act 1985 commercial quantity",
        "Section 20 NDPS cannabis possession",
        "Section 21 NDPS contravention opium",
        "Section 37 NDPS bail restrictions",
        "Section 50 NDPS search personal",
        "conscious possession NDPS drug trafficking",
        
        # Prevention of Corruption
        "Prevention Corruption Act public servant",
        "Section 7 PC Act gratification bribe",
        "Section 13 PC Act criminal misconduct",
        "Section 19 sanction prosecution corruption",
        "disproportionate assets corruption",
        
        # POCSO Act
        "POCSO Act 2012 child sexual abuse",
        "Section 3 POCSO penetrative sexual assault",
        "Section 7 POCSO sexual assault child",
        "Section 11 POCSO sexual harassment",
        "Section 29 POCSO presumption guilt",
        
        # Domestic Violence
        "Domestic Violence Act 2005 protection women",
        "protection order domestic violence",
        "residence order shared household",
        "monetary relief domestic violence",
        
        # SC/ST Act
        "SC ST Prevention Atrocities Act",
        "atrocity against scheduled caste tribe",
        
        # Narcotic & Psychotropic
        "contraband drugs seizure NDPS",
        "drug trafficking conspiracy NDPS",
        
        # Environmental Laws
        "Environment Protection Act 1986",
        "pollution control board orders",
        "NGT National Green Tribunal",
        
        # Consumer Protection
        "Consumer Protection Act deficiency service",
        "unfair trade practice consumer",
        "medical negligence consumer forum",
        
        # RTI Act
        "Right to Information Act 2005",
        "public authority information denial",
        
        # Labour Laws
        "Industrial Disputes Act termination",
        "Employees Provident Fund Act",
        "Minimum Wages Act payment",
        "Payment of Gratuity Act",
        
        # === FAMILY LAW (15 queries) ===
        
        "Hindu Marriage Act divorce cruelty",
        "Section 13 HMA grounds divorce",
        "Section 24 HMA maintenance pendente lite",
        "Section 125 CrPC maintenance wife",
        "Muslim Personal Law talaq divorce",
        "Special Marriage Act inter-religion",
        "Guardians Wards Act custody child",
        "custody minor child welfare",
        "adoption Hindu Adoptions Act",
        "maintenance wife husband Hindu law",
        "partition family property joint",
        "Hindu Succession Act property inheritance",
        "Section 14 absolute property woman",
        "coparcenary daughter equal rights",
        "will testament succession property",
        
        # === PROPERTY & CONTRACT (20 queries) ===
        
        "Transfer Property Act sale immovable",
        "Section 53A part performance contract",
        "Section 54 sale immovable property",
        "adverse possession limitation immovable property",
        "easement right property servient",
        "mortgage redemption foreclosure property",
        "specific performance contract sale",
        "breach contract damages compensation",
        "Indian Contract Act offer acceptance",
        "consideration contract valid essential",
        "void agreement contract law",
        "voidable contract coercion undue influence",
        "frustration contract impossibility performance",
        "quantum meruit contract part performance",
        "indemnity contract loss damage",
        "guarantee contract surety liability",
        "bailment goods custody bailee",
        "pledge goods security debt",
        "agency principal agent authority",
        "partnership agreement profit sharing",
        
        # === SERVICE LAW (15 queries) ===
        
        "Article 311 dismissal removal government servant",
        "departmental inquiry principles natural justice",
        "charge sheet misconduct disciplinary",
        "suspension government employee pending inquiry",
        "termination services probationer",
        "promotion seniority merit government",
        "pension gratuity retirement benefits",
        "compassionate appointment government service",
        "regularization services temporary employee",
        "Central Administrative Tribunal CAT jurisdiction",
        "arbitrary transfer government employee",
        "sexual harassment workplace Vishaka guidelines",
        "leave encashment government employee",
        "pay scale revision government",
        "voluntary retirement scheme VRS",
        
        # === LAND & REVENUE (10 queries) ===
        
        "land acquisition compensation market value",
        "Land Acquisition Act public purpose",
        "enhanced compensation land acquisition solatium",
        "eviction tenant rent control",
        "agricultural land ceiling reforms",
        "mutation land records revenue",
        "khata partition land revenue",
        "encroachment government land removal",
        "land use conversion agricultural",
        "stamp duty registration property",
        
        # === ADDITIONAL HIGH-YIELD QUERIES FOR 10K+ CASES (60 queries) ===
        
        # More Criminal Law
        "bail conditions economic offences",
        "money laundering PMLA attachment",
        "quashing criminal complaint abuse process",
        "plea bargaining criminal procedure",
        "dying declaration evidentiary value",
        "extra judicial confession admissibility",
        "last seen theory circumstantial evidence",
        "motive murder criminal trial",
        "alibi defence criminal case",
        "eyewitness testimony credibility",
        "medical evidence injury report",
        "forensic evidence DNA fingerprint",
        "call records mobile phone evidence",
        "CCTV footage electronic evidence",
        "narco analysis lie detector admissibility",
        
        # More Constitutional Law
        "public interest litigation PIL",
        "contempt of court civil criminal",
        "judicial review administrative action",
        "separation of powers judiciary legislature",
        "appointment judges Supreme Court",
        "collegium system judicial appointments",
        "reservation backward classes OBC",
        "creamy layer reservation exclusion",
        "minority educational institution rights",
        "secularism state religion",
        
        # More Civil & Commercial Law
        "arbitration award challenge Section 34",
        "execution decree civil procedure",
        "attachment property debt recovery",
        "insolvency bankruptcy proceedings NCLT",
        "corporate governance company law",
        "oppression mismanagement minority shareholders",
        "intellectual property patent infringement",
        "trademark copyright violation",
        "defamation civil criminal",
        "injunction restraining order",
        "declaratory suit title property",
        "partition suit family property",
        "adverse possession title limitation",
        "easement prescription property rights",
        "gift deed validity consideration",
        
        # Tax & Economic Laws
        "GST input tax credit",
        "income tax assessment penalty",
        "customs duty import export",
        "benami transaction prohibition",
        "black money undisclosed assets",
        "tax evasion prosecution",
        "appeal income tax tribunal",
        "search seizure tax department",
        
        # More Family & Personal Laws
        "restitution conjugal rights",
        "void voidable marriage",
        "custody grandparents visitation",
        "adoption Hindu law",
        "succession intestate will",
        "probate letters administration",
    ]

def main():
    """Main execution function for massive bulk scraping"""
    print("=" * 80)
    print("MASSIVE BULK SCRAPING: INDIAN KANOON - TARGET 50,000+ CASES")
    print("=" * 80)
    
    # Initialize scraper
    scraper = IndianKanoonScraper()
    
    # Get all queries
    queries = get_massive_queries()
    print(f"\n📋 Total queries: {len(queries)}")
    print(f"🎯 Target: ~{len(queries) * 25:,} cases (25 per query)")
    print(f"⏱️  Estimated time: ~{len(queries) * 2} minutes (2 min per query @ 25 cases)")
    
    # Storage for all cases
    all_cases = []
    seen_urls = set()  # Track unique cases by URL
    
    # Load existing cases if any
    output_file = Path("data/raw/indiankanoon_massive_cases.json")
    start_query_idx = 0
    
    if output_file.exists():
        print(f"\n📂 Loading existing cases from {output_file}...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                all_cases = json.load(f)
                seen_urls = {case['url'] for case in all_cases}
            print(f"✅ Loaded {len(all_cases)} existing cases")
            
            # Calculate which query to start from (assuming 250 cases per query)
            if len(all_cases) > 0:
                start_query_idx = len(all_cases) // 200  # Conservative estimate
                print(f"🔄 Resuming from query {start_query_idx + 1}/{len(queries)}")
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
            print("   Starting fresh...")
            all_cases = []
            seen_urls = set()
    
    start_time = time.time()
    
    # Scrape each query (resume from checkpoint if exists)
    for idx, query in enumerate(queries[start_query_idx:], start_query_idx + 1):
        print(f"\n{'=' * 80}")
        print(f"Query {idx}/{len(queries)}: {query}")
        print(f"{'=' * 80}")
        
        try:
            # Scrape up to 25 cases per query using search_cases method
            cases = scraper.search_cases(query, max_results=25)
            
            # Deduplicate by URL
            new_cases = 0
            for case in cases:
                if case['url'] not in seen_urls:
                    all_cases.append(case)
                    seen_urls.add(case['url'])
                    new_cases += 1
            
            print(f"✅ Query {idx}: {len(cases)} cases scraped, {new_cases} new, {len(all_cases)} total")
            
            # Save progress every 5 queries (more frequent saves)
            if idx % 5 == 0:
                print(f"\n💾 Saving checkpoint at query {idx}...")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_cases, f, indent=2, ensure_ascii=False)
                
                elapsed = time.time() - start_time
                queries_done = idx - start_query_idx
                if queries_done > 0:
                    avg_per_query = elapsed / queries_done
                    remaining = (len(queries) - idx) * avg_per_query
                    print(f"📊 Progress: {idx}/{len(queries)} queries ({idx/len(queries)*100:.1f}%)")
                    print(f"⏱️  Elapsed: {elapsed/60:.1f} min, Estimated remaining: {remaining/60:.1f} min")
                    print(f"📈 Average: {avg_per_query:.1f}s per query, {len(all_cases)/queries_done:.0f} cases per query")
            
            # Respectful delay between queries
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Error on query {idx}: {e}")
            continue
    
    # Final save
    print(f"\n{'=' * 80}")
    print("💾 FINAL SAVE")
    print(f"{'=' * 80}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("✅ SCRAPING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"📊 Total cases scraped: {len(all_cases):,}")
    print(f"🔍 Unique URLs: {len(seen_urls):,}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"📁 Saved to: {output_file}")
    print(f"💾 File size: ~{output_file.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
