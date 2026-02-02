#!/usr/bin/env python3
"""
Generators CLI - Generic Knowledge Graph Generator

Can work with any generator (AI2-THOR, Habitat, Unity, etc.).
Each generator implements the DataSource interface and manages its own config.
"""

import argparse
import sys
from pathlib import Path

# Lazy imports to avoid AI2-THOR import issues at startup
from src.generators.ai2thor.kg.ontology import create_ai2thor_ontology
from src.knowledge_graph.core.builder import KnowledgeGraphBuilder

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Generator CLI")
    parser.add_argument(
        "generator",
        choices=["ai2thor", "habitat", "unity"],  # Add more as implemented
        help="Generator type to use"
    )
    parser.add_argument(
        "--config",
        help="Path to config file (optional, uses generator default)"
    )
    parser.add_argument(
        "--scenes",
        nargs="+", 
        help="Specific scene IDs to process (optional, uses all from config)"
    )
    parser.add_argument(
        "--output",
        default="knowledge_graph.ttl",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["turtle", "json-ld"],
        default="turtle", 
        help="Output format"
    )
    
    args = parser.parse_args()
    
    try:
        # 1. Create data source based on generator type
        if args.generator == "ai2thor":
            # Import AI2-THOR data source only when needed
            from src.generators.ai2thor.kg.data_source import AI2THORDataSource
            
            if args.config:
                data_source = AI2THORDataSource(config_path=args.config)
            else:
                data_source = AI2THORDataSource()  # Use default config
            ontology = create_ai2thor_ontology()
        else:
            raise ValueError(f"Unsupported generator: {args.generator}")
            
        # 2. Create KnowledgeGraphBuilder
        builder = KnowledgeGraphBuilder(ontology)
        
        # 3. Build KG using base methods
        if args.scenes:
            # Process specific scenes
            print(f"Processing {len(args.scenes)} specified scenes: {args.scenes}")
            scene_data = data_source.get_scene_by_id(args.scenes[0])
            print(f"📍 Scene: {scene_data.scene_id}")
            print(f"🔍 Objects found: {len(scene_data.objects)}")
            print(f"🔗 Relationships found: {len(scene_data.relationships)}")
            result = builder.build_from_scene(scene_data)
        else:
            # Process all scenes from config
            available_scenes = data_source.get_available_scenes()
            print(f"Processing {len(available_scenes)} scenes from config: {available_scenes}")
            
            # Show progress for each scene
            total_objects = 0
            total_relationships = 0
            for i, scene_id in enumerate(available_scenes, 1):
                print(f"\n📍 [{i}/{len(available_scenes)}] Processing scene: {scene_id}")
                scene_data = data_source.get_scene_by_id(scene_id)
                print(f"   🔍 Objects: {len(scene_data.objects)}")
                print(f"   🔗 Relationships: {len(scene_data.relationships)}")
                total_objects += len(scene_data.objects)
                total_relationships += len(scene_data.relationships)
            
            print(f"\n📊 Total across all scenes:")
            print(f"   Objects: {total_objects}")
            print(f"   Relationships: {total_relationships}")
            
            result = builder.build_from_source(data_source)
        
        # 4. Save graph
        graph = result.graph
        
        output_format = "turtle" if args.format == "turtle" else "json-ld"
        graph.serialize(destination=args.output, format=output_format)
        
        print(f"✅ Knowledge graph saved to: {args.output}")
        print(f"📊 Total triples: {len(graph)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'data_source' in locals() and hasattr(data_source, 'cleanup'):
            data_source.cleanup()

if __name__ == "__main__":
    main()