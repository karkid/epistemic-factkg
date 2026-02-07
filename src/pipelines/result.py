# src/core/pipeline/result.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineRunResult:
    success: bool
    out_path: str
    format: str
    num_objects: int
    num_relations: int
    total_triples: int
    contexts_processed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PipelineSummary:
    def __init__(self, results: List[PipelineRunResult] = None):
        self.results = results if results is not None else []

    def add_result(self, result: PipelineRunResult):
        self.results.append(result)

    def print_summary(self, logger_func=logger.info, max_items=3, show_lists=False):
        total_objects = sum(r.num_objects for r in self.results)
        total_relations = sum(r.num_relations for r in self.results)
        total_triples = sum(r.total_triples for r in self.results)
        total_receptacles = sum(len(r.contexts_processed) for r in self.results)

        logger_func("="*72)
        logger_func(f"Scenes processed : {len(self.results)}")
        logger_func(f"Total objects     : {total_objects}")
        logger_func(f"Total relations   : {total_relations}")
        logger_func(f"Total triples     : {total_triples}")
        logger_func(f"Total receptacles : {total_receptacles}")
        logger_func(f"Total warnings    : {sum(len(r.warnings) for r in self.results)}")
        logger_func(f"Total errors      : {sum(len(r.errors) for r in self.results)}")
        logger_func("="*72)

        for res in self.results:
            status = "OK" if res.success else "FAILED"

            logger_func("")
            logger_func("=" * 72)
            logger_func(f"Build RDF: {status}")
            logger_func("=" * 72)

            logger_func(f"Output     : {res.out_path}")
            logger_func(f"Format     : {res.format}")
            logger_func(f"Triples    : {res.total_triples}")
            logger_func(f"Objects    : {res.num_objects}")
            logger_func(f"Relations  : {res.num_relations}")

            if res.contexts_processed:
                logger_func(f"Scenes     : {len(res.contexts_processed)}")
                sample = ", ".join(res.contexts_processed[:3])
                logger_func(f"Scene IDs  : {sample}" + (" ..." if len(res.contexts_processed) > 3 else ""))

            logger_func("-" * 72)
            logger_func(f"Warnings   : {len(res.warnings)}")
            logger_func(f"Errors     : {len(res.errors)}")

            if res.warnings:
                logger_func("\nWarnings (sample):")
                for w in res.warnings[:max_items]:
                    logger_func(f"  - {w}")
                if len(res.warnings) > max_items:
                    logger_func(f"  ... ({len(res.warnings) - max_items} more)")
            if res.errors:
                logger_func("\nErrors (sample):")
                for e in res.errors[:max_items]:
                    logger_func(f"  - {e}")
                if len(res.errors) > max_items:
                    logger_func(f"  ... ({len(res.errors) - max_items} more)")

            if show_lists:
                logger_func("\nFULL WARNINGS:")
                for w in res.warnings:
                    logger_func(f"  - {w}")

                logger_func("\nFULL ERRORS:")
                for e in res.errors:
                    logger_func(f"  - {e}")

            logger_func("=" * 72)
            logger_func()