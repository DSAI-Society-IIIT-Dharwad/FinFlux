#!/usr/bin/env python3
"""
POPULATE QUALITY METRICS TABLE FROM EXISTING JSONB DATA
Migrates quality metric data from ai_conversation_messages.model_attribution 
to dedicated ai_conversation_quality_metrics table with full JSONB support
"""
import os
import sys
import json
from datetime import datetime
from datetime import UTC

# Add parent directory to path to import from api/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from api.storage import (
        _supabase_request,
        SUPABASE_CONV_MESSAGES_TABLE,
        SUPABASE_QUALITY_METRICS_TABLE,
        _supabase_conversation_enabled,
        _supabase_quality_enabled
    )
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print(f"Make sure you're running from the finflux directory")
    sys.exit(1)

COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'RESET': '\033[0m',
}

def print_pass(text):
    print(f"{COLORS['GREEN']}[PASS]{COLORS['RESET']} {text}")

def print_fail(text):
    print(f"{COLORS['RED']}[FAIL]{COLORS['RESET']} {text}")

def print_info(text):
    print(f"{COLORS['BLUE']}[INFO]{COLORS['RESET']} {text}")

def print_warn(text):
    print(f"{COLORS['YELLOW']}[WARN]{COLORS['RESET']} {text}")

def main():
    print(f"\n{COLORS['BLUE']}{'='*70}")
    print(" POPULATE QUALITY METRICS TABLE FROM JSONB DATA")
    print(f"{'='*70}{COLORS['RESET']}\n")
    
    if not _supabase_quality_enabled():
        missing = []
        if not os.environ.get("SUPABASE_URL"):
            missing.append("SUPABASE_URL")
        if not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if not os.environ.get("SUPABASE_QUALITY_METRICS_TABLE"):
            missing.append("SUPABASE_QUALITY_METRICS_TABLE")

        print_fail("Supabase quality metrics not configured")
        if missing:
            print_info(f"Missing env vars: {', '.join(missing)}")
        print_info("Set env vars in terminal or create a .env file in finflux root")
        return
    
    try:
        # Step 1: Get all conversations with model_attribution
        print_info("Step 1: Fetching all conversations with quality data...")
        
        messages = _supabase_request(
            "GET",
            f"/rest/v1/{SUPABASE_CONV_MESSAGES_TABLE}",
            params={
                "select": "id,conversation_id,user_id,model_attribution,created_at",
                "order": "created_at.desc",
                "limit": "1000"
            }
        ) or []
        
        if not messages:
            print_warn("No conversations found")
            return
        
        print_pass(f"Found {len(messages)} conversations")
        
        # Step 2: Extract quality metrics JSONB
        print_info("Step 2: Extracting quality metrics from JSONB...")
        
        metrics_to_insert = []
        skipped = 0
        
        for msg in messages:
            model_attr = msg.get('model_attribution') or {}
            if isinstance(model_attr, str):
                try:
                    model_attr = json.loads(model_attr)
                except Exception:
                    model_attr = {}
            
            # Skip if no quality data
            if not model_attr.get('quality_metrics'):
                skipped += 1
                continue
            
            quality_metrics = model_attr.get('quality_metrics', {})
            overall_score = model_attr.get('overall_quality_score')
            quality_tier = model_attr.get('quality_tier', 'ACCEPTABLE')
            
            # Build complete JSONB object
            jsonb_metrics = {
                'quality_metrics': quality_metrics,
                'overall_score': overall_score,
                'tier': quality_tier,
                'timestamp_extracted': datetime.now(UTC).isoformat(),
                'source': 'migration_from_model_attribution'
            }
            
            metrics_to_insert.append({
                'conversation_id': msg.get('conversation_id'),
                'user_id': msg.get('user_id'),
                'overall_quality_score': overall_score,
                'quality_tier': quality_tier,
                'asr_confidence': quality_metrics.get('asr_confidence'),
                'ner_coverage_pct': quality_metrics.get('ner_coverage_pct'),
                'rouge1_recall': quality_metrics.get('rouge1_recall'),
                'entity_alignment_pct': quality_metrics.get('entity_alignment_pct'),
                'language_confidence': quality_metrics.get('language_confidence'),
                'financial_relevance_score': quality_metrics.get('financial_relevance_score'),
                'metadata_jsonb': jsonb_metrics,
                'created_at': msg.get('created_at')
            })
        
        print_pass(f"Extracted {len(metrics_to_insert)} quality metric records")
        print_info(f"Skipped {skipped} records (no quality metrics)")
        
        if not metrics_to_insert:
            print_warn("No quality metrics to insert")
            return
        
        # Step 3: Insert into quality metrics table
        print_info("Step 3: Inserting into ai_conversation_quality_metrics...")
        
        # Insert in batches, with compatibility fallback if metadata_jsonb column is not present yet
        batch_size = 10
        total_inserted = 0
        use_metadata_jsonb = True

        for i in range(0, len(metrics_to_insert), batch_size):
            batch = metrics_to_insert[i:i+batch_size]
            payload_batch = batch

            if not use_metadata_jsonb:
                payload_batch = [{k: v for k, v in row.items() if k != 'metadata_jsonb'} for row in batch]

            try:
                _supabase_request(
                    "POST",
                    f"/rest/v1/{SUPABASE_QUALITY_METRICS_TABLE}",
                    payload=payload_batch
                )
                total_inserted += len(payload_batch)
                print_pass(f"Inserted batch {i//batch_size + 1}: {len(payload_batch)} records")

            except Exception as e:
                msg = str(e)
                if use_metadata_jsonb and 'metadata_jsonb' in msg:
                    print_warn("metadata_jsonb column not found, retrying without metadata_jsonb")
                    use_metadata_jsonb = False
                    payload_batch = [{k: v for k, v in row.items() if k != 'metadata_jsonb'} for row in batch]
                    try:
                        _supabase_request(
                            "POST",
                            f"/rest/v1/{SUPABASE_QUALITY_METRICS_TABLE}",
                            payload=payload_batch
                        )
                        total_inserted += len(payload_batch)
                        print_pass(f"Inserted batch {i//batch_size + 1}: {len(payload_batch)} records")
                    except Exception as retry_err:
                        print_fail(f"Insertion failed: {retry_err}")
                        return
                else:
                    print_fail(f"Insertion failed: {e}")
                    return

        print_pass(f"Successfully inserted {total_inserted} records into quality metrics table")
        
        # Step 4: Verify data
        print_info("Step 4: Verifying inserted data...")
        
        verify = _supabase_request(
            "GET",
            f"/rest/v1/{SUPABASE_QUALITY_METRICS_TABLE}",
            params={
                "select": "id,conversation_id,overall_quality_score,quality_tier",
                "order": "created_at.desc",
                "limit": "10"
            }
        ) or []
        
        print_pass(f"Verification: {len(verify)} records now in quality metrics table")
        
        # Show sample
        if verify:
            print_info("Sample records:")
            for record in verify[:3]:
                conv_id = record.get('conversation_id', '')[:8] if record.get('conversation_id') else 'N/A'
                print(f"  Conversation: {conv_id}...")
                print(f"    Score: {record.get('overall_quality_score', 'N/A')}")
                print(f"    Tier: {record.get('quality_tier')}\n")
        
        # Step 5: Summary
        print(f"\n{COLORS['GREEN']}{'='*70}")
        print(" MIGRATION COMPLETE")
        print(f"{'='*70}{COLORS['RESET']}\n")
        
        print(f"Summary:")
        print(f"  Total Records Processed: {len(messages)}")
        print(f"  Records with Quality Data: {len(metrics_to_insert)}")
        print(f"  Records Skipped: {skipped}")
        print(f"  Successfully Inserted: {total_inserted}")
        print(f"  Current Table Size: {len(verify)}")
        
        print(f"\nQuality Metrics Table Structure:")
        print(f"  - Columns: id, conversation_id, user_id, overall_quality_score,")
        print(f"             quality_tier, asr_confidence, ner_coverage_pct, rouge1_recall,")
        print(f"             entity_alignment_pct, language_confidence,")
        print(f"             financial_relevance_score, metadata_jsonb, created_at")
        print(f"  - JSONB Field: metadata_jsonb (contains all quality metrics)")
        print(f"\nTo verify JSONB content, run:")
        print(f"  SELECT metadata_jsonb FROM ai_conversation_quality_metrics LIMIT 5;\n")
        
    except Exception as e:
        print_fail(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
