"""
Legal Precedent Validator - "Shepardizing" for Indian Case Law

Detects potentially overruled, reversed, or distinguished cases to prevent
citing bad law. Named after Shepard's Citations in US legal research.
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of legal precedent validation"""
    risk_level: str  # "HIGH", "MEDIUM", or "LOW"
    flags: List[str]  # Specific issues found
    warnings: List[str]  # User-facing warning messages
    confidence: float  # Confidence in the assessment (0-1)


class LegalValidator:
    """Validates legal precedents to detect potentially bad law"""
    
    # Keywords indicating a case may have been invalidated
    INVALIDATION_KEYWORDS = {
        "HIGH": [
            "overruled",
            "overturned",
            "reversed",
            "set aside",
            "no longer good law",
            "not followed",
            "expressly disapproved"
        ],
        "MEDIUM": [
            "dissenting",
            "distinguished",
            "questioned",
            "doubted",
            "reconsidered",
            "limited to its facts"
        ]
    }
    
    def __init__(self):
        """Initialize the legal validator"""
        # Compile regex patterns for efficiency
        self.high_risk_pattern = self._compile_pattern(self.INVALIDATION_KEYWORDS["HIGH"])
        self.medium_risk_pattern = self._compile_pattern(self.INVALIDATION_KEYWORDS["MEDIUM"])
    
    def _compile_pattern(self, keywords: List[str]) -> re.Pattern:
        """Compile regex pattern from keyword list"""
        # Use word boundaries to avoid partial matches
        pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def validate_precedent(self, case_text: str) -> ValidationResult:
        """
        Validate a legal precedent by checking for invalidation keywords
        
        Args:
            case_text: Full text of the case
        
        Returns:
            ValidationResult with risk assessment
        """
        flags = []
        warnings = []
        risk_level = "LOW"
        
        # Check for HIGH risk keywords
        high_risk_matches = self.high_risk_pattern.findall(case_text.lower())
        if high_risk_matches:
            risk_level = "HIGH"
            flags.extend([f"Contains '{match}'" for match in set(high_risk_matches)])
            warnings.append(
                "⚠️ WARNING: This case may have been overruled or reversed. "
                "Verify current validity before citing."
            )
        
        # Check for MEDIUM risk keywords (only if not already HIGH)
        if risk_level != "HIGH":
            medium_risk_matches = self.medium_risk_pattern.findall(case_text.lower())
            if medium_risk_matches:
                risk_level = "MEDIUM"
                flags.extend([f"Contains '{match}'" for match in set(medium_risk_matches)])
                warnings.append(
                    "⚠️ CAUTION: This case may have been distinguished or questioned. "
                    "Review subsequent case law."
                )
        
        # Calculate confidence based on match count
        total_matches = len(flags)
        confidence = min(0.9, 0.5 + (total_matches * 0.1))  # 0.5 base + 0.1 per match
        
        return ValidationResult(
            risk_level=risk_level,
            flags=flags,
            warnings=warnings,
            confidence=confidence if flags else 0.7  # Default confidence if no issues
        )
    
    def batch_validate(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate multiple cases and add risk_level to metadata
        
        Args:
            cases: List of case dictionaries with 'text' and 'metadata' keys
        
        Returns:
            List of cases with validation results added to metadata
        """
        print(f"🔍 Validating {len(cases)} cases...")
        
        validated_cases = []
        high_risk_count = 0
        medium_risk_count = 0
        
        for case in cases:
            case_text = case.get('text', '')
            validation = self.validate_precedent(case_text)
            
            # Add validation results to metadata
            if 'metadata' not in case:
                case['metadata'] = {}
            
            case['metadata']['risk_level'] = validation.risk_level
            case['metadata']['validation_flags'] = validation.flags
            case['metadata']['validation_warnings'] = validation.warnings
            case['metadata']['validation_confidence'] = validation.confidence
            
            validated_cases.append(case)
            
            # Track statistics
            if validation.risk_level == "HIGH":
                high_risk_count += 1
            elif validation.risk_level == "MEDIUM":
                medium_risk_count += 1
        
        print(f"✅ Validation complete:")
        print(f"   🔴 HIGH risk: {high_risk_count} cases")
        print(f"   🟡 MEDIUM risk: {medium_risk_count} cases")
        print(f"   🟢 LOW risk: {len(cases) - high_risk_count - medium_risk_count} cases")
        
        return validated_cases
    
    def generate_warning_text(self, validation: ValidationResult) -> str:
        """
        Generate user-facing warning text
        
        Args:
            validation: Validation result
        
        Returns:
            Formatted warning string
        """
        if validation.risk_level == "LOW":
            return ""
        
        warning_parts = []
        
        # Add risk level indicator
        if validation.risk_level == "HIGH":
            warning_parts.append("🚨 HIGH RISK PRECEDENT")
        else:
            warning_parts.append("⚠️ MEDIUM RISK PRECEDENT")
        
        # Add specific flags
        if validation.flags:
            flag_text = ", ".join(validation.flags[:3])  # Limit to 3 flags
            warning_parts.append(f"Issues detected: {flag_text}")
        
        # Add warnings
        warning_parts.extend(validation.warnings)
        
        return "\n".join(warning_parts)
    
    def validate_citation_list(self, citations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Validate a list of case citations (for use in debate system)
        
        Args:
            citations: List of citation dicts with 'case_name' and 'excerpt' keys
        
        Returns:
            List of citations with validation metadata added
        """
        validated_citations = []
        
        for citation in citations:
            # Validate based on excerpt text
            excerpt = citation.get('excerpt', '')
            validation = self.validate_precedent(excerpt)
            
            # Add validation to citation
            citation['risk_level'] = validation.risk_level
            citation['validation_warning'] = self.generate_warning_text(validation)
            
            validated_citations.append(citation)
        
        return validated_citations


# Example usage and testing
if __name__ == "__main__":
    validator = LegalValidator()
    
    # Test case 1: Overruled case
    test_case_1 = """
    In this case, the Court held that... However, this decision was later
    overruled in Subsequent Case v. State (2020) and is no longer good law.
    """
    
    result_1 = validator.validate_precedent(test_case_1)
    print("\n📋 Test Case 1: Overruled case")
    print(f"   Risk Level: {result_1.risk_level}")
    print(f"   Flags: {result_1.flags}")
    print(f"   Warnings: {result_1.warnings}")
    
    # Test case 2: Distinguished case
    test_case_2 = """
    The principle established in this case was distinguished in Later Case v. State
    and limited to its specific facts.
    """
    
    result_2 = validator.validate_precedent(test_case_2)
    print("\n📋 Test Case 2: Distinguished case")
    print(f"   Risk Level: {result_2.risk_level}")
    print(f"   Flags: {result_2.flags}")
    
    # Test case 3: Clean case
    test_case_3 = """
    The Court held that the fundamental right to life under Article 21 includes
    the right to a fair trial. This principle has been consistently followed.
    """
    
    result_3 = validator.validate_precedent(test_case_3)
    print("\n📋 Test Case 3: Clean case")
    print(f"   Risk Level: {result_3.risk_level}")
    print(f"   Flags: {result_3.flags}")
