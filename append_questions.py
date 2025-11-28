import json

new_questions = [
    {
        "id": "q41",
        "question": "What is the 'doctrine of severability'?",
        "ground_truth": "The doctrine of severability (Article 13) provides that if a part of a statute is unconstitutional, only that part is void, provided it can be separated from the valid part. If the valid and invalid parts are inextricably mixed, the whole statute is void.",
        "expected_citations": ["Article 13 Constitution of India", "A.K. Gopalan v. State of Madras"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q42",
        "question": "Can a person be arrested for a non-cognizable offense without a warrant?",
        "ground_truth": "Generally, no. Under Section 41 of CrPC, police cannot arrest without a warrant for non-cognizable offenses. However, Section 42 allows arrest if the person refuses to give their name and residence.",
        "expected_citations": ["Section 41 CrPC", "Section 42 CrPC"],
        "legal_area": "Criminal Procedure"
    },
    {
        "id": "q43",
        "question": "What is 'sedition' under Indian law?",
        "ground_truth": "Sedition was defined under Section 124A of the IPC as bringing or attempting to bring into hatred or contempt, or exciting disaffection towards the Government established by law. (Note: The Supreme Court put this section in abeyance in S.G. Vombatkere v. Union of India, 2022).",
        "expected_citations": ["Section 124A IPC", "Kedar Nath Singh v. State of Bihar"],
        "legal_area": "Criminal Law"
    },
    {
        "id": "q44",
        "question": "Is the right to strike a fundamental right?",
        "ground_truth": "No. The Supreme Court in T.K. Rangarajan v. Government of Tamil Nadu (2003) held that government employees have no fundamental, statutory, or equitable right to strike.",
        "expected_citations": ["T.K. Rangarajan v. Government of Tamil Nadu"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q45",
        "question": "What is the 'Polluter Pays' principle?",
        "ground_truth": "The Polluter Pays principle mandates that the absolute liability for harm to the environment extends not only to compensate the victims of pollution but also the cost of restoring the environmental degradation. It is part of the law of the land under Article 21.",
        "expected_citations": ["Vellore Citizens Welfare Forum v. Union of India", "MC Mehta v. Union of India"],
        "legal_area": "Environmental Law"
    },
    {
        "id": "q46",
        "question": "Can a minor be a partner in a partnership firm?",
        "ground_truth": "A minor cannot be a partner, but under Section 30 of the Indian Partnership Act, 1932, he may be admitted to the benefits of partnership with the consent of all partners.",
        "expected_citations": ["Section 30 Indian Partnership Act"],
        "legal_area": "Partnership Law"
    },
    {
        "id": "q47",
        "question": "What is 'anticipatory breach' of contract?",
        "ground_truth": "Anticipatory breach occurs when a party repudiates the contract before the time for performance has arrived. The aggrieved party can either treat the contract as cancelled and sue for damages immediately or wait until the due date.",
        "expected_citations": ["Section 39 Indian Contract Act", "Hochster v. De La Tour"],
        "legal_area": "Contract Law"
    },
    {
        "id": "q48",
        "question": "Is a confessional statement recorded under TADA/POTA admissible?",
        "ground_truth": "Yes, unlike the general rule under the Evidence Act, confessions made to a police officer of a certain rank were admissible under special acts like TADA (Section 15) and POTA (Section 32), subject to procedural safeguards.",
        "expected_citations": ["Kartar Singh v. State of Punjab", "State v. Nalini"],
        "legal_area": "Criminal Law (Special Acts)"
    },
    {
        "id": "q49",
        "question": "What is the limitation period for filing a consumer complaint?",
        "ground_truth": "Under Section 69 of the Consumer Protection Act, 2019 (and previously Section 24A of the 1986 Act), a complaint must be filed within 2 years from the date on which the cause of action has arisen.",
        "expected_citations": ["Section 69 Consumer Protection Act 2019"],
        "legal_area": "Consumer Law"
    },
    {
        "id": "q50",
        "question": "Can a Muslim woman claim maintenance under Section 125 CrPC?",
        "ground_truth": "Yes. The Supreme Court in Mohd. Ahmed Khan v. Shah Bano Begum (1985) and subsequent cases held that Section 125 CrPC is a secular provision applicable to all religions, including Muslim women, irrespective of personal laws.",
        "expected_citations": ["Mohd. Ahmed Khan v. Shah Bano Begum", "Section 125 CrPC"],
        "legal_area": "Family Law"
    },
    {
        "id": "q51",
        "question": "What is 'Public Interest Litigation' (PIL)?",
        "ground_truth": "PIL is a legal mechanism that allows any public-spirited person or organization to approach the court for the enforcement of public interest or legal rights of those who are unable to approach the court themselves. It relaxes the rule of 'locus standi'.",
        "expected_citations": ["SP Gupta v. Union of India", "Hussainara Khatoon v. Home Secretary"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q52",
        "question": "Is 'Right to Education' a fundamental right?",
        "ground_truth": "Yes. Article 21A, inserted by the 86th Amendment Act, 2002, makes free and compulsory education a fundamental right for all children between the ages of 6 and 14 years.",
        "expected_citations": ["Article 21A Constitution of India", "Unni Krishnan v. State of AP"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q53",
        "question": "What is the 'Doctrine of Eclipse'?",
        "ground_truth": "The Doctrine of Eclipse states that any pre-constitutional law that is inconsistent with fundamental rights is not void ab initio but remains in a dormant or moribund condition (eclipsed). It becomes enforceable again if the inconsistency is removed (e.g., by constitutional amendment).",
        "expected_citations": ["Bhikaji Narain Dhakras v. State of MP", "Article 13(1) Constitution of India"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q54",
        "question": "Can a dying declaration be the sole basis for conviction?",
        "ground_truth": "Yes. A dying declaration (Section 32(1) IEA) can be the sole basis for conviction without corroboration if the court is satisfied that it is true and voluntary.",
        "expected_citations": ["Kushal Rao v. State of Bombay", "Section 32(1) Indian Evidence Act"],
        "legal_area": "Evidence Law"
    },
    {
        "id": "q55",
        "question": "What is 'Hostile Witness'?",
        "ground_truth": "A hostile witness is one who testifies for the party calling them but gives evidence adverse to that party. Under Section 154 of the Evidence Act, the court may permit the party calling the witness to cross-examine them.",
        "expected_citations": ["Section 154 Indian Evidence Act", "Sat Paul v. Delhi Administration"],
        "legal_area": "Evidence Law"
    },
    {
        "id": "q56",
        "question": "What is the difference between 'void' and 'voidable' contract?",
        "ground_truth": "A void contract (Section 2(j)) is not enforceable by law at all. A voidable contract (Section 2(i)) is enforceable by law at the option of one or more of the parties thereto, but not at the option of the other or others (e.g., consent obtained by coercion).",
        "expected_citations": ["Section 2(i) Indian Contract Act", "Section 2(j) Indian Contract Act"],
        "legal_area": "Contract Law"
    },
    {
        "id": "q57",
        "question": "Can a person be punished for an act that was not an offense when committed?",
        "ground_truth": "No. Article 20(1) prohibits 'ex post facto' criminal laws. No person shall be convicted of any offense except for violation of the law in force at the time of the commission of the act.",
        "expected_citations": ["Article 20(1) Constitution of India"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q58",
        "question": "What is 'Constructive Res Judicata'?",
        "ground_truth": "Under Explanation IV to Section 11 CPC, any matter which might and ought to have been made ground of defense or attack in a former suit shall be deemed to have been a matter directly and substantially in issue in such suit.",
        "expected_citations": ["Section 11 Explanation IV CPC", "Forward Construction Co. v. Prabhat Mandal"],
        "legal_area": "Civil Procedure"
    },
    {
        "id": "q59",
        "question": "Is a decree passed by a court without jurisdiction void?",
        "ground_truth": "Yes, a decree passed by a court lacking inherent jurisdiction (subject-matter jurisdiction) is a nullity and its validity can be challenged at any stage, even in execution or collateral proceedings.",
        "expected_citations": ["Kiran Singh v. Chaman Paswan"],
        "legal_area": "Civil Procedure"
    },
    {
        "id": "q60",
        "question": "What is 'Caveat Emptor'?",
        "ground_truth": "Caveat Emptor means 'Let the buyer beware'. Under Section 16 of the Sale of Goods Act, 1930, there is no implied warranty or condition as to the quality or fitness for any particular purpose of goods supplied under a contract of sale, subject to exceptions.",
        "expected_citations": ["Section 16 Sale of Goods Act"],
        "legal_area": "Commercial Law"
    },
    {
        "id": "q61",
        "question": "Can an unborn child be considered a 'person' for transfer of property?",
        "ground_truth": "Yes, property can be transferred for the benefit of an unborn person under Section 13 of the Transfer of Property Act, provided a prior interest is created and the absolute interest is transferred to the unborn person.",
        "expected_citations": ["Section 13 Transfer of Property Act"],
        "legal_area": "Property Law"
    },
    {
        "id": "q62",
        "question": "What is the 'Rule against Perpetuity'?",
        "ground_truth": "Section 14 of the Transfer of Property Act provides that no transfer of property can operate to create an interest which is to take effect after the life-time of one or more persons living at the date of such transfer, and the minority of some person who shall be in existence at the expiration of that period.",
        "expected_citations": ["Section 14 Transfer of Property Act"],
        "legal_area": "Property Law"
    },
    {
        "id": "q63",
        "question": "Is 'Live-in Relationship' recognized in India?",
        "ground_truth": "Yes. The Supreme Court has recognized live-in relationships in the nature of marriage. Women in such relationships are protected under the Protection of Women from Domestic Violence Act, 2005.",
        "expected_citations": ["Indra Sarma v. V.K.V. Sarma", "Velusamy v. D. Patchaiammal"],
        "legal_area": "Family Law"
    },
    {
        "id": "q64",
        "question": "What is the 'Best Interest of the Child' principle?",
        "ground_truth": "In custody battles, the welfare of the child is the paramount consideration, superseding the legal rights of parents. The court decides custody based on what is in the best interest of the child.",
        "expected_citations": ["Gaurav Nagpal v. Sumedha Nagpal", "Section 13 Hindu Minority and Guardianship Act"],
        "legal_area": "Family Law"
    },
    {
        "id": "q65",
        "question": "Can a company buy back its own shares?",
        "ground_truth": "Yes, under Section 68 of the Companies Act, 2013, a company can purchase its own shares (buy-back) out of its free reserves, securities premium account, or proceeds of the issue of any shares or other specified securities, subject to conditions.",
        "expected_citations": ["Section 68 Companies Act 2013"],
        "legal_area": "Corporate Law"
    },
    {
        "id": "q66",
        "question": "What is 'Corporate Veil' and when can it be lifted?",
        "ground_truth": "The Corporate Veil separates the company's personality from its members. It can be lifted (pierced) by courts in cases of fraud, tax evasion, or where the company is a mere cloak or sham (statutory and judicial exceptions).",
        "expected_citations": ["Solomon v. Solomon & Co.", "LIC of India v. Escorts Ltd."],
        "legal_area": "Corporate Law"
    },
    {
        "id": "q67",
        "question": "Is 'Cyber Terrorism' an offense in India?",
        "ground_truth": "Yes, Section 66F of the Information Technology Act, 2000 defines and punishes cyber terrorism (acts denying access to authorized personnel, penetrating secure networks to threaten unity/integrity/sovereignty) with imprisonment up to life.",
        "expected_citations": ["Section 66F Information Technology Act"],
        "legal_area": "Cyber Law"
    },
    {
        "id": "q68",
        "question": "What is the liability of an intermediary (like Facebook/Google) for user content?",
        "ground_truth": "Under Section 79 of the IT Act, intermediaries have 'Safe Harbor' protection from liability for third-party information, provided they observe 'due diligence' and remove content upon receiving a court order or government directive (Shreya Singhal v. Union of India).",
        "expected_citations": ["Section 79 Information Technology Act", "Shreya Singhal v. Union of India"],
        "legal_area": "Cyber Law"
    },
    {
        "id": "q69",
        "question": "Can a trademark be registered for a sound?",
        "ground_truth": "Yes, sound marks can be registered as trademarks in India if they are distinctive and capable of graphical representation (e.g., Yahoo yodel).",
        "expected_citations": ["Trade Marks Act 1999", "Yahoo! Inc. v. Akash Arora"],
        "legal_area": "Intellectual Property Rights"
    },
    {
        "id": "q70",
        "question": "What is the term of copyright for a literary work?",
        "ground_truth": "Under Section 22 of the Copyright Act, 1957, the term of copyright for literary, dramatic, musical, and artistic works is the lifetime of the author plus 60 years from the beginning of the calendar year next following the year in which the author dies.",
        "expected_citations": ["Section 22 Copyright Act 1957"],
        "legal_area": "Intellectual Property Rights"
    },
    {
        "id": "q71",
        "question": "Is 'Patent Evergreening' allowed in India?",
        "ground_truth": "No. Section 3(d) of the Patents Act, 1970 prevents 'evergreening' by stating that the mere discovery of a new form of a known substance which does not result in the enhancement of the known efficacy of that substance is not patentable.",
        "expected_citations": ["Section 3(d) Patents Act 1970", "Novartis v. Union of India"],
        "legal_area": "Intellectual Property Rights"
    },
    {
        "id": "q72",
        "question": "What is 'Plea Bargaining'?",
        "ground_truth": "Plea Bargaining (Chapter XXI-A CrPC) allows an accused to plead guilty in exchange for a lesser sentence. It applies to offenses punishable with imprisonment up to 7 years, excluding offenses against women/children or socio-economic offenses.",
        "expected_citations": ["Chapter XXI-A CrPC", "Section 265A CrPC"],
        "legal_area": "Criminal Procedure"
    },
    {
        "id": "q73",
        "question": "Can a private person arrest someone?",
        "ground_truth": "Yes. Under Section 43 of CrPC, a private person can arrest any person who in their presence commits a non-bailable and cognizable offense, or is a proclaimed offender. They must hand over the arrested person to the police immediately.",
        "expected_citations": ["Section 43 CrPC"],
        "legal_area": "Criminal Procedure"
    },
    {
        "id": "q74",
        "question": "What is 'Injunction'?",
        "ground_truth": "An injunction is a judicial order restraining a person from doing or continuing an act (preventive) or commanding them to do a particular act (mandatory). It is governed by the Specific Relief Act, 1963 and CPC.",
        "expected_citations": ["Specific Relief Act 1963", "Order 39 CPC"],
        "legal_area": "Civil Procedure"
    },
    {
        "id": "q75",
        "question": "What is 'Force Majeure'?",
        "ground_truth": "Force Majeure is a clause in contracts that frees both parties from liability or obligation when an extraordinary event or circumstance beyond their control (Act of God, war, etc.) prevents one or both from fulfilling their obligations. Section 56 of the Contract Act (Frustration) is related.",
        "expected_citations": ["Section 56 Indian Contract Act", "Satyabrata Ghose v. Mugneeram Bangur"],
        "legal_area": "Contract Law"
    },
    {
        "id": "q76",
        "question": "Can an arbitrator's award be challenged in court?",
        "ground_truth": "Yes, under Section 34 of the Arbitration and Conciliation Act, 1996, an arbitral award can be set aside by a court on limited grounds like incapacity of a party, invalid arbitration agreement, lack of notice, or conflict with public policy.",
        "expected_citations": ["Section 34 Arbitration and Conciliation Act"],
        "legal_area": "Arbitration Law"
    },
    {
        "id": "q77",
        "question": "What is 'Indemnity'?",
        "ground_truth": "A contract of indemnity (Section 124 Contract Act) is a contract by which one party promises to save the other from loss caused to him by the conduct of the promisor himself, or by the conduct of any other person.",
        "expected_citations": ["Section 124 Indian Contract Act"],
        "legal_area": "Contract Law"
    },
    {
        "id": "q78",
        "question": "Is 'Adultery' a crime in India?",
        "ground_truth": "No. In Joseph Shine v. Union of India (2018), the Supreme Court struck down Section 497 of the IPC, decriminalizing adultery. However, it remains a valid ground for divorce in civil law.",
        "expected_citations": ["Joseph Shine v. Union of India", "Section 497 IPC (Struck Down)"],
        "legal_area": "Criminal Law / Constitutional Law"
    },
    {
        "id": "q79",
        "question": "What is the 'Golden Triangle' of the Indian Constitution?",
        "ground_truth": "Articles 14 (Equality), 19 (Freedoms), and 21 (Life and Liberty) form the Golden Triangle. They are mutually reinforcing and must be read together. A law depriving a person of personal liberty must satisfy the requirements of Articles 14 and 19 as well.",
        "expected_citations": ["Maneka Gandhi v. Union of India"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q80",
        "question": "Can a Governor pardon a death sentence?",
        "ground_truth": "Yes, but with limitations. Under Article 161, the Governor has the power to grant pardons, etc. However, the Supreme Court has clarified that while the Governor can pardon, the President's power under Article 72 is wider (especially regarding Court Martial). Recent judgments suggest Governor can pardon death sentence (Perarivalan case).",
        "expected_citations": ["Article 161 Constitution of India", "Article 72 Constitution of India", "A.G. Perarivalan v. State"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q81",
        "question": "What is 'Judicial Review'?",
        "ground_truth": "Judicial Review is the power of the judiciary (Supreme Court and High Courts) to examine the constitutionality of legislative enactments and executive orders. It is a basic feature of the Constitution.",
        "expected_citations": ["Article 13", "Article 32", "Article 226", "L. Chandra Kumar v. Union of India"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q82",
        "question": "What is 'Habeas Corpus'?",
        "ground_truth": "Habeas Corpus ('to have the body') is a writ issued by the court directing a person or authority who has detained another person to produce the detainee before the court to test the legality of the detention.",
        "expected_citations": ["Article 32", "Article 226", "ADM Jabalpur v. Shivkant Shukla"],
        "legal_area": "Constitutional Law"
    },
    {
        "id": "q83",
        "question": "Can a woman be arrested after sunset?",
        "ground_truth": "Generally, no. Section 46(4) of CrPC states that no woman shall be arrested after sunset and before sunrise. In exceptional circumstances, a woman police officer can arrest with the prior permission of the Judicial Magistrate First Class.",
        "expected_citations": ["Section 46(4) CrPC"],
        "legal_area": "Criminal Procedure"
    },
    {
        "id": "q84",
        "question": "What is 'Bail'?",
        "ground_truth": "Bail is the release of an accused person from legal custody, usually on the condition that they will appear in court when required. It is a matter of right in bailable offenses (Section 436 CrPC) and discretionary in non-bailable offenses (Section 437/439 CrPC).",
        "expected_citations": ["Section 436 CrPC", "Section 437 CrPC", "State of Rajasthan v. Balchand"],
        "legal_area": "Criminal Procedure"
    },
    {
        "id": "q85",
        "question": "What is 'Cheating'?",
        "ground_truth": "Cheating (Section 415 IPC) involves deceiving a person and fraudulently or dishonestly inducing them to deliver property or consent to retain property, or to do/omit something they wouldn't otherwise do.",
        "expected_citations": ["Section 415 IPC", "Section 420 IPC"],
        "legal_area": "Criminal Law"
    },
    {
        "id": "q86",
        "question": "What is 'Criminal Breach of Trust'?",
        "ground_truth": "Criminal Breach of Trust (Section 405 IPC) occurs when a person entrusted with property or dominion over property dishonestly misappropriates or converts it to their own use.",
        "expected_citations": ["Section 405 IPC", "Section 406 IPC"],
        "legal_area": "Criminal Law"
    },
    {
        "id": "q87",
        "question": "What is 'Sexual Harassment' at workplace?",
        "ground_truth": "Sexual Harassment is defined under the POSH Act, 2013 (and Vishaka Guidelines). It includes physical contact, demand for sexual favors, sexually colored remarks, showing pornography, or unwelcome physical/verbal/non-verbal conduct of sexual nature.",
        "expected_citations": ["POSH Act 2013", "Vishaka v. State of Rajasthan"],
        "legal_area": "Labor/Women's Law"
    },
    {
        "id": "q88",
        "question": "What is 'Gratuity'?",
        "ground_truth": "Gratuity is a monetary benefit given by an employer to an employee upon retirement or exit after at least 5 years of continuous service. It is governed by the Payment of Gratuity Act, 1972.",
        "expected_citations": ["Payment of Gratuity Act 1972"],
        "legal_area": "Labor Law"
    },
    {
        "id": "q89",
        "question": "Can an employee be fired without notice?",
        "ground_truth": "Generally, no. Labor laws (like Industrial Disputes Act) require notice or pay in lieu of notice for retrenchment. However, in cases of proven misconduct, dismissal might be immediate after a due disciplinary inquiry.",
        "expected_citations": ["Industrial Disputes Act 1947"],
        "legal_area": "Labor Law"
    },
    {
        "id": "q90",
        "question": "What is 'RTI'?",
        "ground_truth": "Right to Information (RTI) Act, 2005 empowers citizens to request information from public authorities, promoting transparency and accountability in government.",
        "expected_citations": ["Right to Information Act 2005"],
        "legal_area": "Administrative Law"
    },
    {
        "id": "q91",
        "question": "Is 'Deficiency in Service' covered under Consumer Protection Act?",
        "ground_truth": "Yes. Section 2(11) of the Consumer Protection Act, 2019 defines 'deficiency' as any fault, imperfection, shortcoming or inadequacy in the quality, nature and manner of performance required to be maintained.",
        "expected_citations": ["Section 2(11) Consumer Protection Act 2019", "Indian Medical Association v. V.P. Shantha"],
        "legal_area": "Consumer Law"
    },
    {
        "id": "q92",
        "question": "What is 'Unfair Trade Practice'?",
        "ground_truth": "Unfair Trade Practice (Section 2(47) CPA 2019) includes false representation, misleading advertisements, hoarding, refusal to issue bill, etc., adopted to promote sale/use/supply of goods or services.",
        "expected_citations": ["Section 2(47) Consumer Protection Act 2019"],
        "legal_area": "Consumer Law"
    },
    {
        "id": "q93",
        "question": "What is 'GST'?",
        "ground_truth": "Goods and Services Tax (GST) is a comprehensive indirect tax levied on the supply of goods and services. It replaced multiple indirect taxes like VAT, excise duty, service tax, etc. (101st Constitutional Amendment).",
        "expected_citations": ["101st Constitutional Amendment Act", "CGST Act 2017"],
        "legal_area": "Tax Law"
    },
    {
        "id": "q94",
        "question": "What is 'Money Laundering'?",
        "ground_truth": "Money Laundering involves projecting proceeds of crime as untainted property. It is an offense under the Prevention of Money Laundering Act, 2002 (PMLA).",
        "expected_citations": ["Prevention of Money Laundering Act 2002", "Vijay Madanlal Choudhary v. Union of India"],
        "legal_area": "Criminal Law (Economic Offenses)"
    },
    {
        "id": "q95",
        "question": "What is 'Benami Transaction'?",
        "ground_truth": "A Benami transaction is one where property is transferred to one person for a consideration paid or provided by another person. It is prohibited under the Prohibition of Benami Property Transactions Act, 1988.",
        "expected_citations": ["Prohibition of Benami Property Transactions Act 1988"],
        "legal_area": "Property/Tax Law"
    },
    {
        "id": "q96",
        "question": "Can a person execute a 'Living Will'?",
        "ground_truth": "Yes. The Supreme Court in Common Cause v. Union of India (2018) upheld the validity of Advance Medical Directives (Living Wills) for passive euthanasia.",
        "expected_citations": ["Common Cause v. Union of India"],
        "legal_area": "Constitutional/Health Law"
    },
    {
        "id": "q97",
        "question": "What is 'Lok Adalat'?",
        "ground_truth": "Lok Adalat ('People's Court') is an alternative dispute resolution mechanism organized under the Legal Services Authorities Act, 1987, to settle disputes amicably and quickly.",
        "expected_citations": ["Legal Services Authorities Act 1987"],
        "legal_area": "ADR"
    },
    {
        "id": "q98",
        "question": "Is 'Triple Talaq' a crime?",
        "ground_truth": "Yes. The Muslim Women (Protection of Rights on Marriage) Act, 2019 makes the pronouncement of Triple Talaq (Talaq-e-Biddat) a cognizable and non-bailable offense punishable with imprisonment up to 3 years.",
        "expected_citations": ["Muslim Women (Protection of Rights on Marriage) Act 2019"],
        "legal_area": "Family Law"
    },
    {
        "id": "q99",
        "question": "What is the age of consent for sex in India?",
        "ground_truth": "The age of consent is 18 years. Sexual intercourse with a woman below 18 years is rape, regardless of consent (POCSO Act and IPC).",
        "expected_citations": ["POCSO Act 2012", "Independent Thought v. Union of India"],
        "legal_area": "Criminal Law"
    },
    {
        "id": "q100",
        "question": "Can a transgender person identify as 'Third Gender'?",
        "ground_truth": "Yes. In NALSA v. Union of India (2014), the Supreme Court recognized transgender persons as a 'third gender' and affirmed their fundamental rights under the Constitution.",
        "expected_citations": ["NALSA v. Union of India"],
        "legal_area": "Constitutional Law"
    }
]

# Append to existing file
with open('data/benchmark_questions.json', 'r') as f:
    existing = json.load(f)

combined = existing + new_questions

with open('data/benchmark_questions.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Successfully created {len(combined)} questions.")
