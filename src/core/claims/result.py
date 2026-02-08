from dataclasses import dataclass, field
from typing import Dict, List
from src.core.claims.types import ClaimCorpus


@dataclass
class ClaimGenerationStats:
    """Advanced statistics for claim generation process."""
    
    # Core claim counts
    supported: int = 0
    refuted: int = 0
    skipped_empty: int = 0
    corruption_hits_true: int = 0
    
    # Detailed tracking
    total_processed_triples: int = 0
    failed_corruptions: int = 0
    successful_corruptions: int = 0
    duplicate_claims_filtered: int = 0
    
    # Corruption strategy breakdown
    corruption_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Object type analysis
    object_types_processed: Dict[str, int] = field(default_factory=dict)
    receptacle_mappings_used: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    processing_time_seconds: float = 0.0
    average_time_per_triple: float = 0.0
    
    # Final corpus
    corpus: ClaimCorpus = field(default_factory=lambda: ClaimCorpus([]))
    
    def add_corruption_type(self, corruption_type: str) -> None:
        """Track corruption types used."""
        if corruption_type in self.corruption_by_type:
            self.corruption_by_type[corruption_type] += 1
        else:
            self.corruption_by_type[corruption_type] = 1
    
    def add_object_type(self, object_type: str) -> None:
        """Track object types processed."""
        if object_type in self.object_types_processed:
            self.object_types_processed[object_type] += 1
        else:
            self.object_types_processed[object_type] = 1
    
    def add_receptacle_mapping(self, receptacle: str) -> None:
        """Track receptacle mappings used in corruption."""
        if receptacle in self.receptacle_mappings_used:
            self.receptacle_mappings_used[receptacle] += 1
        else:
            self.receptacle_mappings_used[receptacle] = 1
    
    def finalize_timing(self) -> None:
        """Calculate final timing metrics."""
        if self.total_processed_triples > 0 and self.processing_time_seconds > 0:
            self.average_time_per_triple = self.processing_time_seconds / self.total_processed_triples
    
    @property
    def total_claims_generated(self) -> int:
        """Total claims generated (supported + refuted)."""
        return self.supported + self.refuted
    
    @property
    def corruption_success_rate(self) -> float:
        """Percentage of successful corruptions."""
        total_attempts = self.successful_corruptions + self.failed_corruptions
        return (self.successful_corruptions / total_attempts * 100) if total_attempts > 0 else 0.0
    
    @property
    def claim_generation_rate(self) -> float:
        """Percentage of processed triples that resulted in claims."""
        return (self.total_claims_generated / self.total_processed_triples * 100) if self.total_processed_triples > 0 else 0.0
    
    @property
    def supported_to_refuted_ratio(self) -> float:
        """Ratio of supported to refuted claims."""
        return (self.supported / self.refuted) if self.refuted > 0 else float('inf')


@dataclass
class ClaimGenerationStatsSummary:
    """Summary and reporting functionality for claim generation statistics."""
    
    stats: ClaimGenerationStats
    
    def print_summary(self) -> None:
        """Print comprehensive statistics summary."""
        print("\n" + "="*60)
        print("CLAIM GENERATION STATISTICS SUMMARY")
        print("="*60)
        
        # Core metrics
        print(f"\n📊 CORE METRICS:")
        print(f"   Total triples processed: {self.stats.total_processed_triples:,}")
        print(f"   Total claims generated:  {self.stats.total_claims_generated:,}")
        print(f"   • Supported claims:      {self.stats.supported:,}")
        print(f"   • Refuted claims:        {self.stats.refuted:,}")
        print(f"   Skipped (empty):         {self.stats.skipped_empty:,}")
        print(f"   Claim generation rate:   {self.stats.claim_generation_rate:.1f}%")
        
        # Corruption metrics
        print(f"\n🔄 CORRUPTION ANALYSIS:")
        print(f"   Successful corruptions:  {self.stats.successful_corruptions:,}")
        print(f"   Failed corruptions:      {self.stats.failed_corruptions:,}")
        print(f"   Corruption success rate: {self.stats.corruption_success_rate:.1f}%")
        print(f"   Corruption hits (true):  {self.stats.corruption_hits_true:,}")
        
        # Balance analysis
        print(f"\n⚖️  CLAIM BALANCE:")
        if self.stats.refuted > 0:
            print(f"   Supported:Refuted ratio: {self.stats.supported_to_refuted_ratio:.2f}:1")
        else:
            print(f"   Supported:Refuted ratio: {self.stats.supported}:0 (no refuted claims)")
        
        if self.stats.duplicate_claims_filtered > 0:
            print(f"   Duplicates filtered:     {self.stats.duplicate_claims_filtered:,}")
        
        # Corruption strategy breakdown
        if self.stats.corruption_by_type:
            print(f"\n🎯 CORRUPTION STRATEGIES:")
            for corruption_type, count in sorted(self.stats.corruption_by_type.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / sum(self.stats.corruption_by_type.values())) * 100
                print(f"   • {corruption_type}: {count:,} ({percentage:.1f}%)")
        
        # Top object types
        if self.stats.object_types_processed:
            print(f"\n🎨 TOP OBJECT TYPES PROCESSED:")
            sorted_objects = sorted(self.stats.object_types_processed.items(), key=lambda x: x[1], reverse=True)
            for obj_type, count in sorted_objects[:10]:  # Top 10
                percentage = (count / sum(self.stats.object_types_processed.values())) * 100
                print(f"   • {obj_type}: {count:,} ({percentage:.1f}%)")
            
            if len(sorted_objects) > 10:
                remaining = len(sorted_objects) - 10
                print(f"   ... and {remaining} other object types")
        
        # Top receptacle mappings
        if self.stats.receptacle_mappings_used:
            print(f"\n🗄️  TOP RECEPTACLE MAPPINGS USED:")
            sorted_receptacles = sorted(self.stats.receptacle_mappings_used.items(), key=lambda x: x[1], reverse=True)
            for receptacle, count in sorted_receptacles[:5]:  # Top 5
                percentage = (count / sum(self.stats.receptacle_mappings_used.values())) * 100
                print(f"   • {receptacle}: {count:,} ({percentage:.1f}%)")
        
        # Performance metrics
        if self.stats.processing_time_seconds > 0:
            print(f"\n⏱️  PERFORMANCE:")
            print(f"   Total processing time:   {self.stats.processing_time_seconds:.2f} seconds")
            if self.stats.average_time_per_triple > 0:
                print(f"   Average time per triple: {self.stats.average_time_per_triple*1000:.2f} ms")
            print(f"   Processing rate:         {self.stats.total_processed_triples/self.stats.processing_time_seconds:.1f} triples/sec")
        
        # Final corpus info
        print(f"\n📝 FINAL CORPUS:")
        print(f"   Claims in corpus:        {len(self.stats.corpus.claims):,}")
        
        print("="*60)
        print("✅ Claim generation completed successfully!")
        print("="*60)