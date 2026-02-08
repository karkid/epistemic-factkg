#!/usr/bin/env python3
"""
Pipeline function for validating claim JSONL files with advanced quality checks.
"""

import json
import time
from pathlib import Path

from src.core.claims.claim_validator import AdvancedClaimValidator, AdvancedClaimValidationSummary
from src.core.claims.types import ClaimCorpus
from src.utils.logger import get_logger
from src.utils.exceptions import ValidationError

logger = get_logger(__name__)


def validate_claims(
    claims_file: str,
    ttl_path: str = None,  # For future RDF-based validation
    verbose: bool = False
) -> AdvancedClaimValidationSummary:
    """
    Validate claims from a JSONL file using advanced validation checks.
    
    Args:
        claims_file: Path to the claims JSONL file
        ttl_path: Path to TTL file (for future RDF-based validation)
        verbose: Enable verbose logging
        
    Returns:
        AdvancedClaimValidationSummary with comprehensive validation results
    """
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting advanced validation of claims file: {claims_file}")
    
    claims_path = Path(claims_file)
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_file}")
    
    # Initialize advanced validator and summary
    validator = AdvancedClaimValidator()
    summary = AdvancedClaimValidationSummary()
    
    # Store claim texts for detailed reporting
    claim_texts = {}
    claim_data = {}
    
    try:
        # Read and validate claims
        with open(claims_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON
                    claim_json = json.loads(line)
                    claim_id = claim_json.get('id', f'line_{line_num}')
                    claim_text = claim_json.get('claim', '')
                    
                    # Store claim text and data for reporting
                    claim_texts[claim_id] = claim_text
                    claim_data[claim_id] = claim_json
                    
                    # Perform advanced validation
                    validation_result = validator.validate_claim_advanced(claim_json, claim_id)
                    summary.add_result(validation_result)
                    
                    if verbose and line_num % 100 == 0:
                        logger.info(f"Validated {line_num} claims...")
                        
                except json.JSONDecodeError as e:
                    # Create a result for JSON decode errors
                    from src.core.claims.claim_validator import ClaimValidationResult
                    error_result = ClaimValidationResult(claim_id=f'line_{line_num}', is_valid=False)
                    error_result.add_issue("schema", "error", f"Invalid JSON: {str(e)}")
                    summary.add_result(error_result)
                    claim_texts[f'line_{line_num}'] = line[:100] + "..." if len(line) > 100 else line
                    
                    if verbose:
                        logger.warning(f"Line {line_num}: JSON decode error - {str(e)}")
                
                except Exception as e:
                    # Create a result for unexpected errors
                    from src.core.claims.claim_validator import ClaimValidationResult
                    error_result = ClaimValidationResult(claim_id=f'line_{line_num}', is_valid=False)
                    error_result.add_issue("system", "error", f"Unexpected error: {str(e)}")
                    summary.add_result(error_result)
                    claim_texts[f'line_{line_num}'] = "Error reading claim"
                    
                    if verbose:
                        logger.error(f"Line {line_num}: Unexpected error - {str(e)}")
    
    except Exception as e:
        logger.error(f"Error reading claims file: {str(e)}")
        # Return empty summary with file read error
        from src.core.claims.claim_validator import ClaimValidationResult
        error_result = ClaimValidationResult(claim_id='file_error', is_valid=False)
        error_result.add_issue("system", "error", f"Error reading file: {str(e)}")
        summary.add_result(error_result)
        return summary
    
    # Add claim texts and data to summary for detailed reporting
    summary.claim_texts = claim_texts
    summary.claim_data = claim_data
    
    processing_time = time.time() - start_time
    
    if verbose:
        valid_count = sum(1 for r in summary.results if r.is_valid)
        total_count = len(summary.results)
        logger.info(f"Advanced validation completed: {valid_count}/{total_count} claims valid ({processing_time:.2f}s)")
    
    return summary