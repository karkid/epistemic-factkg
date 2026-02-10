import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from urllib.parse import urlparse
from jsonschema import validate, ValidationError
from datetime import datetime

from src.utils.exceptions import ValidationError as CustomValidationError
from src.core.claims.claim_schema import CLAIM_SCHEMA


@dataclass
class ClaimValidationIssue:
    """Represents a validation issue found in a claim."""
    category: str  # schema, semantic, quality, consistency, etc.
    severity: str  # error, warning, info
    message: str
    field: Optional[str] = None
    value: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ClaimValidationResult:
    """Detailed validation result for a single claim."""
    claim_id: str
    is_valid: bool
    issues: List[ClaimValidationIssue] = field(default_factory=list)
    quality_score: float = 0.0  # 0.0 to 1.0
    processing_time_ms: float = 0.0
    
    # Category-specific results
    schema_valid: bool = True
    semantic_valid: bool = True
    quality_valid: bool = True
    consistency_valid: bool = True
    
    def add_issue(self, category: str, severity: str, message: str, field: str = None, value: str = None, suggestion: str = None):
        """Add a validation issue."""
        issue = ClaimValidationIssue(category, severity, message, field, value, suggestion)
        self.issues.append(issue)
        
        # Update validity flags
        if severity == "error":
            self.is_valid = False
            if category == "schema":
                self.schema_valid = False
            elif category == "semantic":
                self.semantic_valid = False
            elif category == "quality":
                self.quality_valid = False
            elif category == "consistency":
                self.consistency_valid = False


@dataclass
class AdvancedClaimValidationSummary:
    """Comprehensive summary of claim validation results."""
    results: List[ClaimValidationResult] = field(default_factory=list)
    claim_texts: Dict[str, str] = field(default_factory=dict)
    
    def add_result(self, result: ClaimValidationResult):
        """Add a validation result."""
        self.results.append(result)
    
    def print_summary(self):
        """Print comprehensive validation summary."""
        if not self.results:
            print("No validation results to display.")
            return
        
        # Calculate aggregate statistics
        total_claims = len(self.results)
        valid_claims = sum(1 for r in self.results if r.is_valid)
        invalid_claims = total_claims - valid_claims
        
        # Category statistics
        schema_valid = sum(1 for r in self.results if r.schema_valid)
        semantic_valid = sum(1 for r in self.results if r.semantic_valid)
        quality_valid = sum(1 for r in self.results if r.quality_valid)
        consistency_valid = sum(1 for r in self.results if r.consistency_valid)
        
        # Quality statistics
        avg_quality_score = sum(r.quality_score for r in self.results) / total_claims if total_claims > 0 else 0
        total_issues = sum(len(r.issues) for r in self.results)
        avg_processing_time = sum(r.processing_time_ms for r in self.results) / total_claims if total_claims > 0 else 0
        
        # Issue analysis
        issue_by_category = {}
        issue_by_severity = {}
        for result in self.results:
            for issue in result.issues:
                issue_by_category[issue.category] = issue_by_category.get(issue.category, 0) + 1
                issue_by_severity[issue.severity] = issue_by_severity.get(issue.severity, 0) + 1
        
        print("\n" + "="*70)
        print("ADVANCED CLAIMS VALIDATION SUMMARY")
        print("="*70)
        
        # Overall results
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total claims validated:  {total_claims:,}")
        print(f"   Valid claims:           {valid_claims:,} ({valid_claims/total_claims*100:.1f}%)")
        print(f"   Invalid claims:         {invalid_claims:,} ({invalid_claims/total_claims*100:.1f}%)")
        print(f"   Average quality score:  {avg_quality_score:.3f}/1.000")
        
        # Category breakdown
        print(f"\n🔍 VALIDATION CATEGORIES:")
        print(f"   Schema validation:      {schema_valid:,}/{total_claims:,} ({schema_valid/total_claims*100:.1f}%)")
        print(f"   Semantic validation:    {semantic_valid:,}/{total_claims:,} ({semantic_valid/total_claims*100:.1f}%)")
        print(f"   Quality validation:     {quality_valid:,}/{total_claims:,} ({quality_valid/total_claims*100:.1f}%)")
        print(f"   Consistency validation: {consistency_valid:,}/{total_claims:,} ({consistency_valid/total_claims*100:.1f}%)")
        
        # Issue analysis
        if total_issues > 0:
            print(f"\n⚠️  ISSUE ANALYSIS:")
            print(f"   Total issues found:     {total_issues:,}")
            print(f"   Average issues per claim: {total_issues/total_claims:.1f}")
            
            if issue_by_severity:
                print(f"\n   By severity:")
                for severity, count in sorted(issue_by_severity.items(), key=lambda x: {'error': 3, 'warning': 2, 'info': 1}.get(x[0], 0), reverse=True):
                    emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(severity, "•")
                    print(f"     {emoji} {severity.title()}: {count:,} ({count/total_issues*100:.1f}%)")
            
            if issue_by_category:
                print(f"\n   By category:")
                for category, count in sorted(issue_by_category.items(), key=lambda x: x[1], reverse=True):
                    print(f"     • {category.title()}: {count:,} ({count/total_issues*100:.1f}%)")
            
            # Detailed issue breakdown
            print(f"\n   🔍 DETAILED ISSUE BREAKDOWN:")
            issue_details = {}
            for result in self.results:
                for issue in result.issues:
                    key = f"[{issue.severity.upper()}] {issue.category}: {issue.message}"
                    if key not in issue_details:
                        issue_details[key] = {
                            'count': 0,
                            'examples': [],
                            'claim_ids': []
                        }
                    issue_details[key]['count'] += 1
                    if len(issue_details[key]['examples']) < 3:  # Keep first 3 examples
                        # Get the claim text for context
                        claim_text = ""
                        for res in self.results:
                            if res.claim_id == result.claim_id:
                                # Try to get claim text from the original validation
                                break
                        issue_details[key]['examples'].append({
                            'claim_id': result.claim_id,
                            'field': issue.field,
                            'suggestion': issue.suggestion
                        })
                    if len(issue_details[key]['claim_ids']) < 10:  # Keep first 10 IDs
                        issue_details[key]['claim_ids'].append(result.claim_id)
            
            # Show top 5 most common issues with details
            sorted_issues = sorted(issue_details.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            for issue_text, details in sorted_issues:
                print(f"\n     📋 {details['count']:3d}x {issue_text}")
                
                # Show example claim IDs with actual claim text
                if details['claim_ids']:
                    print(f"         🏷️  Example claims:")
                    example_ids = details['claim_ids'][:3]  # Show first 3 examples
                    for i, claim_id in enumerate(example_ids, 1):
                        claim_text = self.claim_texts.get(claim_id, "Text not available")
                        # Truncate long texts
                        display_text = claim_text[:80] + "..." if len(claim_text) > 80 else claim_text
                        print(f"         {i}. {claim_id}: \"{display_text}\"")
                    
                    if len(details['claim_ids']) > 3:
                        remaining = len(details['claim_ids']) - 3
                        print(f"         ... and {remaining} more claims with this issue")
                
                # Show field and suggestion if available
                if details['examples']:
                    first_example = details['examples'][0]
                    if first_example['field']:
                        print(f"         📍 Field: {first_example['field']}")
                    if first_example['suggestion']:
                        print(f"         💡 Suggestion: {first_example['suggestion']}")
            
            if len(sorted_issues) == 5 and len(issue_details) > 5:
                remaining = len(issue_details) - 5
                print(f"\n     ... and {remaining} other issue types")
            
            # Summary of claims with issues
            claims_with_issues = set()
            for result in self.results:
                if result.issues:
                    claims_with_issues.add(result.claim_id)
            
            if claims_with_issues:
                print(f"\n   📊 AFFECTED CLAIMS:")
                print(f"     Claims with issues: {len(claims_with_issues):,}/{total_claims:,} ({len(claims_with_issues)/total_claims*100:.1f}%)")
                
                # Show first few affected claim IDs
                sample_ids = sorted(list(claims_with_issues))[:5]
                print(f"     Sample affected claims: {', '.join(sample_ids)}")
                if len(claims_with_issues) > 5:
                    remaining = len(claims_with_issues) - 5
                    print(f"     ... and {remaining} more")
        
        # Performance metrics
        if avg_processing_time > 0:
            print(f"\n⏱️  PERFORMANCE:")
            print(f"   Average validation time: {avg_processing_time:.2f} ms/claim")
            total_time = sum(r.processing_time_ms for r in self.results) / 1000
            print(f"   Total validation time:   {total_time:.2f} seconds")
        
        # Quality distribution
        quality_ranges = {"Excellent (0.9-1.0)": 0, "Good (0.7-0.9)": 0, "Fair (0.5-0.7)": 0, "Poor (0.0-0.5)": 0}
        for result in self.results:
            if result.quality_score >= 0.9:
                quality_ranges["Excellent (0.9-1.0)"] += 1
            elif result.quality_score >= 0.7:
                quality_ranges["Good (0.7-0.9)"] += 1
            elif result.quality_score >= 0.5:
                quality_ranges["Fair (0.5-0.7)"] += 1
            else:
                quality_ranges["Poor (0.0-0.5)"] += 1
        
        print(f"\n📈 QUALITY DISTRIBUTION:")
        for range_name, count in quality_ranges.items():
            percentage = count / total_claims * 100 if total_claims > 0 else 0
            print(f"   {range_name}: {count:,} ({percentage:.1f}%)")
        
        # Claims breakdown by type and label
        self._print_claims_breakdown()
        
        # Final status
        if invalid_claims == 0:
            print("\n" + "="*70)
            print("✅ ALL CLAIMS PASSED ADVANCED VALIDATION!")
        else:
            print("\n" + "="*70)
            print(f"⚠️  {invalid_claims:,} CLAIMS FAILED ADVANCED VALIDATION")
            
            if total_issues > 0:
                print(f"\nTop issues to address:")
                sorted_categories = sorted(issue_by_category.items(), key=lambda x: x[1], reverse=True)[:3]
                for i, (category, count) in enumerate(sorted_categories, 1):
                    print(f"   {i}. Fix {category} issues ({count:,} occurrences)")
        
        print("="*70)

    def _print_claims_breakdown(self):
        """Print breakdown of claims by structural reasoning and label."""
        print(f"\n📊 CLAIMS BREAKDOWN BY TYPE:")
        
        # Count claims by structural reasoning and label
        breakdown = {}
        for result in self.results:
            # Extract structural reasoning and label from stored claim data
            structural = "unknown"
            label = "unknown" 
            
            # Try to get from claim_data storage
            if hasattr(self, 'claim_data') and result.claim_id in self.claim_data:
                claim_json = self.claim_data[result.claim_id]
                
                # Handle different schema formats
                # New format: reasoning.structural
                if 'reasoning' in claim_json and isinstance(claim_json['reasoning'], dict) and 'structural' in claim_json['reasoning']:
                    structural = claim_json['reasoning']['structural']
                # Old format: structural_reasoning
                elif 'structural_reasoning' in claim_json:
                    structural = claim_json['structural_reasoning']
                
                # Handle label (normalize case)
                if 'label' in claim_json:
                    label = claim_json['label'].lower()
            
            key = f"{structural}_{label}"
            if key not in breakdown:
                breakdown[key] = 0
            breakdown[key] += 1
        
        # Sort and display the breakdown
        if breakdown and breakdown != {"unknown_unknown": len(self.results)}:
            # Group by structural type
            structural_types = {}
            for key, count in breakdown.items():
                structural, label = key.split('_', 1)
                if structural not in structural_types:
                    structural_types[structural] = {}
                structural_types[structural][label] = count
            
            # Display organized breakdown
            for structural in sorted(structural_types.keys()):
                labels = structural_types[structural]
                total_for_type = sum(labels.values())
                print(f"   📋 {structural.upper()}:")
                print(f"      Total: {total_for_type:,}")
                
                # Show supported and refuted counts
                supported = labels.get('supported', 0)
                refuted = labels.get('refuted', 0)
                if supported > 0:
                    print(f"      - Supported: {supported:,}")
                if refuted > 0:
                    print(f"      - Refuted: {refuted:,}")
                
                # Handle other labels
                other_labels = {k: v for k, v in labels.items() if k not in ['supported', 'refuted']}
                if other_labels:
                    for other_label, count in sorted(other_labels.items()):
                        if other_label != 'unknown':
                            print(f"      - {other_label.title()}: {count:,}")
                
                if supported + refuted + sum(other_labels.values()) < total_for_type:
                    other = total_for_type - supported - refuted - sum(other_labels.values())
                    print(f"      - Other: {other:,}")
        else:
            print("   Unable to determine claim type breakdown (metadata not available)")


class AdvancedClaimValidator:
    """Advanced claim validator with comprehensive quality checks."""
    
    def __init__(self):
        self.uri_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]*$')
        self.sentence_endings = re.compile(r'[.!?]$')
        self.common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'will', 'would', 'can', 'could', 'should', 'must', 'not'}
    
    def validate_claim_advanced(self, claim_json: dict, claim_id: str = None) -> ClaimValidationResult:
        """Perform comprehensive validation of a claim."""
        start_time = time.time()
        
        # Extract claim ID
        actual_claim_id = claim_id or claim_json.get('id', 'unknown')
        result = ClaimValidationResult(claim_id=actual_claim_id, is_valid=True)
        
        try:
            # 1. Schema validation
            self._validate_schema(claim_json, result)
            
            # 2. Semantic validation
            self._validate_semantics(claim_json, result)
            
            # 3. Quality validation
            self._validate_quality(claim_json, result)
            
            # 4. Consistency validation
            self._validate_consistency(claim_json, result)
            
            # 5. Calculate quality score
            result.quality_score = self._calculate_quality_score(claim_json, result)
            
        except Exception as e:
            result.add_issue("system", "error", f"Validation system error: {str(e)}")
        
        # Record processing time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_schema(self, claim_json: dict, result: ClaimValidationResult):
        """Validate against JSON schema."""
        try:
            validate(instance=claim_json, schema=CLAIM_SCHEMA)
        except ValidationError as e:
            result.add_issue("schema", "error", f"Schema validation failed: {e.message}", field=str(e.path) if e.path else None)
    
    def _validate_semantics(self, claim_json: dict, result: ClaimValidationResult):
        """Enhanced semantic validation."""
        try:
            triples = claim_json.get("claim_triples", [])
            reasoning = claim_json.get("reasoning", {}).get("structural", "")
            claim_text = claim_json.get("claim", "")
            
            # Basic reasoning checks
            if reasoning == "one-hop" and len(triples) != 1:
                result.add_issue("semantic", "error", f"One-hop claims must have exactly 1 triple, found {len(triples)}", "claim_triples")
            
            if reasoning == "conjunction" and len(triples) < 2:
                result.add_issue("semantic", "error", f"Conjunction claims must have >=2 triples, found {len(triples)}", "claim_triples")
            
            if reasoning == "negation" and not ("not" in claim_text.lower() or "false" in claim_text.lower() or "n't" in claim_text.lower()):
                result.add_issue("semantic", "warning", "Negation claims should contain negation words ('not', 'false', \"n't\")", "claim", suggestion="Add explicit negation to claim text")
            
            # Advanced semantic checks
            if reasoning == "conjunction":
                if " and " not in claim_text.lower() and " also " not in claim_text.lower():
                    result.add_issue("semantic", "warning", "Conjunction claims should explicitly connect concepts with 'and' or 'also'", "claim")
            
            # Triple validation - handle both array and object formats
            for i, triple in enumerate(triples):
                if isinstance(triple, list) and len(triple) == 3:
                    # Array format: [subject, predicate, object]
                    s, p, o = triple
                    # URI validation for subject and predicate
                    for field_name, uri in [('subject', s), ('predicate', p)]:
                        if not self._is_valid_uri(str(uri)):
                            result.add_issue("semantic", "warning", f"Triple {i+1} {field_name} may not be a valid URI: {uri}", f"claim_triples[{i}]")
                elif isinstance(triple, dict) and all(k in triple for k in ['s', 'p', 'o']):
                    # Object format: {s: ..., p: ..., o: ...}
                    for field in ['s', 'p']:
                        uri = str(triple[field])
                        if not self._is_valid_uri(uri):
                            result.add_issue("semantic", "warning", f"Triple {i+1} {field} may not be a valid URI: {uri}", f"claim_triples[{i}].{field}")
                else:
                    result.add_issue("semantic", "error", f"Triple {i+1} is malformed - expected array [s,p,o] or object {{s,p,o}}", f"claim_triples[{i}]")
            
            # Receptacle mapping validation for refuted claims
            self._validate_receptacle_mapping(claim_json, result)
            
        except Exception as e:
            result.add_issue("semantic", "error", f"Semantic validation error: {str(e)}")
    
    def _validate_quality(self, claim_json: dict, result: ClaimValidationResult):
        """Validate claim quality and readability."""
        try:
            claim_text = claim_json.get("claim", "").strip()
            
            # Basic text quality
            if not claim_text:
                result.add_issue("quality", "error", "Claim text is empty", "claim")
                return
            
            if len(claim_text) < 10:
                result.add_issue("quality", "warning", f"Claim text is very short ({len(claim_text)} chars)", "claim", suggestion="Ensure claim is descriptive enough")
            
            if len(claim_text) > 200:
                result.add_issue("quality", "warning", f"Claim text is very long ({len(claim_text)} chars)", "claim", suggestion="Consider simplifying the claim")
            
            # Sentence structure
            if not self.sentence_endings.search(claim_text):
                result.add_issue("quality", "warning", "Claim doesn't end with proper punctuation", "claim", suggestion="Add period, exclamation mark, or question mark")
            
            if not claim_text[0].isupper():
                result.add_issue("quality", "warning", "Claim doesn't start with capital letter", "claim")
            
            # Word analysis
            words = claim_text.lower().split()
            if len(words) < 3:
                result.add_issue("quality", "warning", f"Claim has very few words ({len(words)})", "claim")
            
            # Check for meaningful content
            meaningful_words = [w for w in words if w not in self.common_words and len(w) > 2]
            if len(meaningful_words) < 2:
                result.add_issue("quality", "warning", "Claim may lack meaningful content", "claim")
            
            # Check for repetition
            if len(set(words)) < len(words) * 0.5:  # More than 50% repeated words (was 70%)
                result.add_issue("quality", "info", "Claim contains significant word repetition", "claim")
            
            # Check for obvious word duplication (e.g., "the the", "is is")
            for i in range(len(words) - 1):
                if words[i] == words[i + 1] and len(words[i]) > 2:  # Skip short words like "a a"
                    result.add_issue("quality", "warning", f"Claim contains duplicated word: '{words[i]} {words[i+1]}'", "claim", suggestion="Remove duplicate word")
                    break  # Only report first occurrence

            # Advanced NLG quality checks
            self._validate_nlg_quality(claim_json, result, claim_text, words)
            
        except Exception as e:
            result.add_issue("quality", "error", f"Quality validation error: {str(e)}")
    
    def _validate_nlg_quality(self, claim_json: dict, result: ClaimValidationResult, claim_text: str, words: List[str]):
        """Validate Natural Language Generation quality."""
        try:
            claim_triples = claim_json.get("claim_triples", [])
            
            # Check for temperature-related NLG issues
            if any("temperature" in str(triple).lower() for triple in claim_triples):
                # Temperature claims should have proper formatting
                if " is temperature" in claim_text.lower():
                    result.add_issue("quality", "error", "Temperature claim has malformed NLG: should be 'is hot/cold/at room temperature' not 'is temperature'", "claim", 
                                   suggestion="Use proper temperature values like 'is hot', 'is cold', or 'is at room temperature'")
                elif not any(temp_phrase in claim_text.lower() for temp_phrase in ["at room temperature", "hot", "cold"]):
                    result.add_issue("quality", "warning", "Temperature claim should specify actual temperature value", "claim")
                    
            # Check for boolean property NLG issues  
            boolean_props = ["openable", "toggleable", "pickable", "movable", "cookable", "sliceable", "breakable", "dirtyable"]
            for prop in boolean_props:
                if any(prop in str(triple).lower() for triple in claim_triples):
                    # Boolean properties should not show "true" or "false" in text
                    if " is true" in claim_text.lower() or " is false" in claim_text.lower():
                        result.add_issue("quality", "error", f"Boolean property claim shows raw boolean value instead of proper NLG", "claim",
                                       suggestion=f"Use '{prop}' or 'not {prop}' instead of 'true' or 'false'")
                    elif " true" in claim_text.lower() or " false" in claim_text.lower():
                        result.add_issue("quality", "warning", f"Boolean claim may contain raw boolean values", "claim")
                        
            # Check for URI fragments in claim text (should be human-readable)
            if "http://" in claim_text or "https://" in claim_text:
                result.add_issue("quality", "error", "Claim text contains raw URIs instead of human-readable names", "claim",
                               suggestion="Convert URIs to human-readable entity names")
                               
            # Check for entity ID patterns that weren't converted properly
            if "|" in claim_text and "%" in claim_text:
                result.add_issue("quality", "warning", "Claim may contain unconverted entity IDs", "claim")
                
            # Check for awkward phrasing patterns
            awkward_patterns = [
                ("is temperature", "should specify temperature value (hot/cold/room temperature)"),
                ("temperature.", "should be more specific about temperature"),
                ("is true.", "boolean properties should not show 'true'"),
                ("is false.", "boolean properties should not show 'false'"),
                ("the the ", "duplicate articles"),
                ("a a ", "duplicate articles"),
            ]
            
            for pattern, suggestion in awkward_patterns:
                if pattern in claim_text.lower():
                    result.add_issue("quality", "warning", f"Awkward phrasing detected: '{pattern}'", "claim", suggestion=suggestion)
                    
        except Exception as e:
            result.add_issue("quality", "error", f"NLG quality validation error: {str(e)}")

    def _validate_consistency(self, claim_json: dict, result: ClaimValidationResult):
        """Validate internal consistency."""
        try:
            label = claim_json.get("label", "")
            claim_text = claim_json.get("claim", "")
            
            # Handle different evidence formats
            evidence = claim_json.get("evidence", {})
            if isinstance(evidence, dict):
                evidence_triples = evidence.get("triples", []) or evidence.get("evidence_triples", [])
            else:
                evidence_triples = []
            
            claim_triples = claim_json.get("claim_triples", [])
            
            # Label consistency
            if label not in ["supported", "refuted"]:
                result.add_issue("consistency", "error", f"Invalid label: {label}", "label")
            
            # Evidence-claim consistency
            if len(evidence_triples) == 0:
                result.add_issue("consistency", "info", "No evidence triples provided", "evidence.triples")
            
            # Metadata consistency
            meta = claim_json.get("meta", {})
            if meta.get("created_utc"):
                try:
                    datetime.fromisoformat(meta["created_utc"].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    result.add_issue("consistency", "warning", "Invalid timestamp format", "meta.created_utc")
            
            # Context consistency - handle different context formats
            context = claim_json.get("context", {})
            context_id = context.get("id") or context.get("context_id")
            if not context_id:
                result.add_issue("consistency", "info", "Missing context ID", "context.id or context.context_id")
            
            # Evidence-claim triple consistency based on label
            if evidence_triples and claim_triples:
                # Normalize triples for comparison (handle both array and object formats)
                normalized_claim_triples = self._normalize_triples(claim_triples)
                normalized_evidence_triples = self._normalize_triples(evidence_triples)
                
                triples_match = normalized_claim_triples == normalized_evidence_triples
                
                if label == "supported":
                    # Supported claims should have matching evidence and claim triples
                    if not triples_match:
                        result.add_issue("consistency", "error", "Supported claim has mismatched evidence and claim triples", "evidence.triples vs claim_triples")
                elif label == "refuted":
                    # Refuted (corrupted) claims should have different evidence and claim triples
                    if triples_match:
                        result.add_issue("consistency", "error", "Refuted claim has identical evidence and claim triples - corruption not applied", "evidence.triples vs claim_triples")
            
        except Exception as e:
            result.add_issue("consistency", "error", f"Consistency validation error: {str(e)}")
    
    def _validate_receptacle_mapping(self, claim_json: dict, result: ClaimValidationResult):
        """Validate receptacle mapping coverage for spatial corruption."""
        try:
            label = claim_json.get("label", "").lower()
            claim_triples = claim_json.get("claim_triples", [])
            evidence_triples = claim_json.get("evidence", {}).get("evidence_triples", [])
            
            # Only check refuted claims with spatial relationships
            if label != "refuted":
                return
                
            for i, triple in enumerate(claim_triples):
                if not isinstance(triple, list) or len(triple) != 3:
                    continue
                    
                s, p, o = triple
                
                # Check for spatial relationships that should use receptacle corruption
                if "onTopOf" in str(p) or "inside" in str(p) or "near" in str(p):
                    # Extract object type from subject
                    if "/" in str(s):
                        obj_type = str(s).split("/")[-1].split("%7C")[0] if "%7C" in str(s) else str(s).split("/")[-1]
                        
                        # Check if object has receptacle mapping
                        if self._has_receptacle_mapping(obj_type):
                            result.add_issue("semantic", "info", f"Object '{obj_type}' has receptacle mapping available for corruption", f"claim_triples[{i}]")
                        else:
                            result.add_issue("semantic", "warning", f"Object '{obj_type}' missing from receptacle mapping rules - spatial corruption not possible", f"claim_triples[{i}]", suggestion=f"Add '{obj_type}' to SEMANTIC_RECEPTACLE_RULES")
                            
        except Exception as e:
            result.add_issue("semantic", "error", f"Receptacle mapping validation error: {str(e)}")
    
    def _has_receptacle_mapping(self, obj_type: str) -> bool:
        """Check if an object type has receptacle mapping rules."""
        try:
            # Import the semantic rules to check coverage
            from src.adapters.ai2thor.semantics.semantic_rules import get_preferred_receptacles
            return len(get_preferred_receptacles(obj_type)) > 0
        except ImportError:
            return False
    
    def _calculate_quality_score(self, claim_json: dict, result: ClaimValidationResult) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        score = 1.0
        
        # Deduct points for issues
        for issue in result.issues:
            if issue.severity == "error":
                score -= 0.3
            elif issue.severity == "warning":
                score -= 0.1
            elif issue.severity == "info":
                score -= 0.02
        
        # Bonus points for good practices
        claim_text = claim_json.get("claim", "")
        if claim_text:
            if 10 <= len(claim_text) <= 100:  # Good length
                score += 0.05
            if self.sentence_endings.search(claim_text):  # Proper punctuation
                score += 0.02
            if len(claim_text.split()) >= 5:  # Sufficient detail
                score += 0.03
        
        return max(0.0, min(1.0, score))
    
    def _is_valid_uri(self, uri_string: str) -> bool:
        """Check if string is a valid URI."""
        try:
            if not uri_string:
                return False
            return self.uri_pattern.match(uri_string) is not None
        except:
            return False
    
    def _normalize_triples(self, triples: List) -> List[Tuple[str, str, str]]:
        """Normalize triples to a consistent format for comparison."""
        normalized = []
        for triple in triples:
            if isinstance(triple, list) and len(triple) == 3:
                # Array format: [subject, predicate, object]
                normalized.append((str(triple[0]), str(triple[1]), str(triple[2])))
            elif isinstance(triple, dict) and all(k in triple for k in ['s', 'p', 'o']):
                # Object format: {s: ..., p: ..., o: ...}
                normalized.append((str(triple['s']), str(triple['p']), str(triple['o'])))
        return sorted(normalized)  # Sort for consistent comparison


# Legacy functions for backward compatibility
def semantic_checks(claim: dict):
    """Legacy semantic checks function."""
    validator = AdvancedClaimValidator()
    result = validator.validate_claim_advanced(claim)
    
    # Extract error messages for legacy compatibility
    error_messages = [issue.message for issue in result.issues if issue.severity == "error" and issue.category == "semantic"]
    if error_messages:
        raise ValueError(error_messages[0])


def validate_claim(claim_json: dict) -> bool:
    """Legacy validate claim function."""
    validator = AdvancedClaimValidator()
    result = validator.validate_claim_advanced(claim_json)
    
    if not result.is_valid:
        error_issues = [issue for issue in result.issues if issue.severity == "error"]
        if error_issues:
            error_message = error_issues[0].message
            if error_issues[0].category == "schema":
                raise CustomValidationError(f"Claim JSON failed validation: {error_message}")
            else:
                raise ValueError(error_message)
    
    return True
