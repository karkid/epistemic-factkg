#!/usr/bin/env python3
"""
CLI for validating semantic claims JSONL files against schema and semantic rules.
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipelines.validate_claims import validate_claims
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Validate semantic claims JSONL files against schema and semantic rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "ttl_path",
        help="Path to the TTL file (currently unused, reserved for future RDF-based validation)"
    )
    
    parser.add_argument(
        "--claims-file", "-c",
        required=True,
        help="Path to the claims JSONL file to validate"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save validation reports (optional, future feature)"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop validation on first error (future feature)"
    )
    
    parser.add_argument(
        "--export-errors",
        action="store_true",
        help="Export validation errors to a separate file"
    )
    
    parser.add_argument(
        "--export-detailed-report",
        action="store_true", 
        help="Export detailed validation report with all issues and claim texts"
    )
    
    args = parser.parse_args()
    
    # Enable debug logging if verbose
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.disabled = False
    
    try:
        # Validate claims file path
        claims_path = Path(args.claims_file)
        if not claims_path.exists():
            logger.error(f"Claims file not found: {args.claims_file}")
            sys.exit(1)
        
        # Validate TTL file path (if provided)
        ttl_path = Path(args.ttl_path)
        if not ttl_path.exists():
            logger.warning(f"TTL file not found: {args.ttl_path} (validation will proceed without RDF context)")
        
        logger.info(f"Starting claims validation...")
        logger.info(f"Claims file: {args.claims_file}")
        if ttl_path.exists():
            logger.info(f"TTL file: {args.ttl_path}")
        
        # Run validation
        summary = validate_claims(
            claims_file=args.claims_file,
            ttl_path=args.ttl_path if ttl_path.exists() else None,
            verbose=args.verbose
        )
        
        # Display comprehensive summary
        summary.print_summary()
        
        # Export detailed reports if requested
        if args.export_detailed_report or args.export_errors:
            issues_to_export = []
            for result in summary.results:
                if result.issues and (not args.export_errors or not result.is_valid):
                    for issue in result.issues:
                        issues_to_export.append({
                            'claim_id': result.claim_id,
                            'claim_text': summary.claim_texts.get(result.claim_id, 'N/A'),
                            'is_valid': result.is_valid,
                            'quality_score': result.quality_score,
                            'issue_category': issue.category,
                            'issue_severity': issue.severity,
                            'issue_message': issue.message,
                            'issue_field': issue.field,
                            'suggestion': issue.suggestion
                        })
            
            if issues_to_export:
                output_dir = Path(args.output_dir) if args.output_dir else Path('.')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if args.export_detailed_report:
                    report_file = output_dir / "detailed_validation_report.jsonl"
                    with open(report_file, 'w') as f:
                        for issue_data in issues_to_export:
                            f.write(json.dumps(issue_data) + '\n')
                    logger.info(f"📁 Detailed validation report exported to: {report_file}")
                
                if args.export_errors and any(not summary.results[i].is_valid for i in range(len(summary.results))):
                    error_file = output_dir / "validation_errors.jsonl"
                    error_issues = [issue for issue in issues_to_export if issue['issue_severity'] == 'error']
                    with open(error_file, 'w') as f:
                        for issue_data in error_issues:
                            f.write(json.dumps(issue_data) + '\n')
                    logger.info(f"❌ Validation errors exported to: {error_file}")
            else:
                logger.info("No issues to export.")
        
        # Exit with appropriate code
        valid_count = sum(1 for r in summary.results if r.is_valid)
        total_count = len(summary.results)
        invalid_count = total_count - valid_count
        
        if invalid_count == 0:
            logger.info("✅ All claims passed advanced validation!")
            sys.exit(0)
        else:
            logger.error(f"❌ Advanced validation failed: {invalid_count}/{total_count} claims failed validation")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()