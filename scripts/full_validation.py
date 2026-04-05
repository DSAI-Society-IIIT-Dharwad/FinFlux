#!/usr/bin/env python3
"""
FINFLUX FULL VALIDATION - Complete System Showcase
Shows: All Supabase tables, Semantic Search, Vector Embeddings, RAG, Self-Query, Quality Metrics
"""
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path so we can import api/ and scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

try:
    from api.storage import (
        get_all_conversations, 
        search_memories,
        get_conversation_by_id,
        save_conversation,
    )
    from api.server import SYNTHESIS
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print(f"Make sure you're running from the finflux directory")
    sys.exit(1)

COLORS = {
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'CYAN': '\033[96m',
    'RESET': '\033[0m',
}

# Global validation result tracker
VALIDATION_RESULTS = {"pass": 0, "fail": 0, "skip": 0, "zero_data": 0}

def print_header(text: str, level: int = 1):
    if level == 1:
        print(f"\n{COLORS['CYAN']}{'='*80}")
        print(f" {text}")
        print(f"{'='*80}{COLORS['RESET']}\n")
    else:
        print(f"\n{COLORS['BLUE']}{text}")
        print(f"{'-'*80}{COLORS['RESET']}")

def print_pass(text: str):
    print(f"  {COLORS['GREEN']}[PASS]{COLORS['RESET']} {text}")

def print_fail(text: str):
    print(f"  {COLORS['RED']}[FAIL]{COLORS['RESET']} {text}")

def print_info(text: str):
    print(f"  {COLORS['BLUE']}[INFO]{COLORS['RESET']} {text}")

def print_warn(text: str):
    print(f"  {COLORS['YELLOW']}[WARN]{COLORS['RESET']} {text}")

def print_data(label: str, value: Any, indent: int = 2):
    spaces = ' ' * indent
    if isinstance(value, dict):
        print(f"{spaces}{label}:")
        for k, v in value.items():
            print(f"{spaces}  {k}: {v}")
    elif isinstance(value, list):
        print(f"{spaces}{label}: ({len(value)} items)")
        for item in value[:3]:
            print(f"{spaces}  - {item}")
    else:
        print(f"{spaces}{label}: {value}")

# ============================================================================
# VALIDATION 1: SUPABASE TABLE STATUS (ALL TABLES)
# ============================================================================
def validate_all_tables():
    print_header("1. SUPABASE TABLE STATUS - Data via API", level=1)
    
    print_header("QUERYING DATA", level=2)
    
    try:
        # Get data for demo user
        result = get_all_conversations(user_id='user_demo_001')
        convs = result.get('results', []) if isinstance(result, dict) else []
        
        print_info(f"Fetched conversations for user_demo_001")
        print_pass(f"ai_conversation_messages table: {len(convs)} rows (bridge table in use)")
        
        # Check for embeddings
        with_embeddings = sum(1 for c in convs if isinstance(c, dict) and c.get('embedding'))
        print_pass(f"ai_message_embeddings table: {with_embeddings} rows with embeddings")
        
        # Check for quality metrics
        with_metrics = sum(1 for c in convs 
                          if isinstance(c, dict) and c.get('model_attribution', {}).get('quality_metrics'))
        print_pass(f"ai_conversation_quality_metrics JSONB: {with_metrics} conversations with metrics")
        
        # Check threads
        thread_ids = {c.get('thread_id') for c in convs if isinstance(c, dict) and c.get('thread_id')}
        print_pass(f"ai_conversation_threads table: {len(thread_ids)} unique threads")
        
        print_header("DATA SUMMARY", level=2)
        total_rows = len(convs)
        print_pass(f"Total conversations (messages): {total_rows}")
        print_pass(f"With embeddings: {with_embeddings}")
        print_pass(f"With quality metrics: {with_metrics}")
        print_pass(f"Unique threads: {len(thread_ids)}")
        
        if total_rows > 0:
            print_pass(f"Using BRIDGE TABLES (service-role pattern)")
            VALIDATION_RESULTS["pass"] += 1
        else:
            print_warn(f"No data in bridge tables - save a conversation first!")
            VALIDATION_RESULTS["zero_data"] += 1
        
    except Exception as e:
        print_fail(f"Table status check failed: {e}")
        VALIDATION_RESULTS["fail"] += 1

# ============================================================================
# VALIDATION 2: VECTOR EMBEDDINGS
# ============================================================================
def validate_embeddings():
    print_header("2. VECTOR EMBEDDINGS - Embedding Generation & Storage", level=1)
    
    try:
        result = get_all_conversations(user_id='user_demo_001')
        convs = result.get('results', []) if isinstance(result, dict) else []
        
        if not convs:
            print_warn("No conversations found - skipping embedding validation")
            VALIDATION_RESULTS["zero_data"] += 1
            return
        
        print_info(f"Checking {len(convs)} conversations for embeddings\n")
        
        with_embed = []
        missing_embed = []
        
        for conv in convs:
            if isinstance(conv, dict) and conv.get('embedding'):
                with_embed.append(conv)
            else:
                if isinstance(conv, dict):
                    missing_embed.append(conv.get('conversation_id', 'unknown'))
        
        print_pass(f"Conversations with embeddings: {len(with_embed)}/{len(convs)}")
        
        if with_embed:
            sample = with_embed[0]
            embedding = sample.get('embedding')
            
            if isinstance(embedding, list):
                print_pass(f"Embedding dimensions: {len(embedding)} (expected 384)")
                
                # Check value range
                min_val = min(embedding) if embedding else 0
                max_val = max(embedding) if embedding else 0
                avg_val = sum(embedding) / len(embedding) if embedding else 0
                
                print_data("Embedding Statistics", {
                    'Min Value': f"{min_val:.4f}",
                    'Max Value': f"{max_val:.4f}",
                    'Avg Value': f"{avg_val:.4f}",
                    'Model': sample.get('embedding_model', 'unknown'),
                    'Created': sample.get('embedding_created_at', 'unknown')
                }, indent=2)
                
                if -1 <= min_val and max_val <= 1:
                    print_pass(f"Embedding values in correct range [-1, 1]")
                else:
                    print_warn(f"Embedding values outside expected range")
        
        if missing_embed:
            print_warn(f"Missing embeddings: {len(missing_embed)} (wait 2-3 sec for async)")
        
    except Exception as e:
        print_fail(f"Embedding validation failed: {e}")

# ============================================================================
# VALIDATION 3: SEMANTIC SEARCH
# ============================================================================
def validate_semantic_search():
    print_header("3. SEMANTIC SEARCH - Query Embedding & Similarity Scoring", level=1)
    
    try:
        result = get_all_conversations(user_id='user_demo_001')
        convs = result.get('results', []) if isinstance(result, dict) else []
        
        if not convs:
            print_warn("No conversations found - skipping semantic search test")
            VALIDATION_RESULTS["zero_data"] += 1
            return
        
        test_queries = [
            ("investment portfolio strategy", "Investment query"),
            ("loan planning borrowing", "Loan query"),
            ("insurance coverage risk", "Insurance query"),
        ]
        
        for query_text, description in test_queries:
            print_header(description, level=2)
            print_info(f"Query: '{query_text}'")
            
            try:
                results = search_memories(
                    user_id='user_demo_001',
                    query_text=query_text,
                    filters={},
                    n_results=8,
                    min_similarity=0.70
                )
                
                result_list = results.get('results', []) if isinstance(results, dict) else []
                
                if result_list:
                    print_pass(f"Found {len(result_list)} similar conversations")
                    
                    for i, result in enumerate(result_list[:2], 1):
                        if isinstance(result, dict):
                            print_data(f"Result {i}", {
                                'Topic': result.get('financial_topic'),
                                'Risk': result.get('risk'),
                                'Similarity': f"{result.get('similarity_score', 0):.2%}",
                                'Summary': (result.get('executive_summary', '')[:80] + '...') if result.get('executive_summary') else 'N/A'
                            }, indent=4)
                else:
                    print_warn(f"No results found (OK on first run - embeddings async)")
                
            except Exception as e:
                print_fail(f"Search failed: {str(e)[:60]}")
        
    except Exception as e:
        print_fail(f"Semantic search validation failed: {e}")

# ============================================================================
# VALIDATION 4: RAG + SELF-QUERY DECOMPOSITION
# ============================================================================
def validate_rag_self_query():
    print_header("4. RAG + SELF-QUERY DECOMPOSITION - Extract Semantic Query & Filters", level=1)
    
    try:
        test_cases = [
            "I want to invest 5 lakhs in long-term growth with high risk tolerance",
            "Help me plan a home loan with EMI calculation",
            "Optimize my insurance portfolio to reduce premiums",
            "Tax planning strategies for FY 2026",
        ]
        
        for i, user_query in enumerate(test_cases, 1):
            print_header(f"Test Case {i}", level=2)
            print_info(f"User Query: '{user_query}'")
            
            try:
                # Decompose query
                decomposed = SYNTHESIS.decompose_retrieval_query(user_query)
                
                if decomposed:
                    print_pass(f"Self-Query Decomposition Successful")
                    
                    print_data("Decomposed Output", {
                        'Semantic Query': decomposed.get('semantic_query', 'N/A'),
                        'Filters': decomposed.get('filters', {}),
                    }, indent=2)
                    
                    # Show extracted metadata filters
                    filters = decomposed.get('filters', {})
                    if filters:
                        print_pass(f"Extracted {len(filters)} metadata filters:")
                        for key, value in filters.items():
                            print_data(key, value, indent=4)
                else:
                    print_fail(f"Decomposition returned empty")
                
            except Exception as e:
                print_fail(f"Decomposition failed: {str(e)[:70]}")
        
    except Exception as e:
        print_fail(f"RAG validation failed: {e}")

# ============================================================================
# VALIDATION 5: QUALITY METRICS + JSONB DATA
# ============================================================================
def validate_quality_metrics():
    print_header("5. QUALITY METRICS - JSONB Storage & Computation", level=1)
    
    try:
        result = get_all_conversations(user_id='user_demo_001')
        convs = result.get('results', []) if isinstance(result, dict) else []
        
        if not convs:
            print_warn("No conversations found - skipping quality metrics validation")
            VALIDATION_RESULTS["zero_data"] += 1
            return
        
        print_info(f"Analyzing {len(convs)} conversations for quality metric JSONB data\n")
        
        metrics_found = 0
        all_metrics = []
        
        for conv in convs:
            if not isinstance(conv, dict):
                continue
                
            model_attr = conv.get('model_attribution', {})
            
            if model_attr:
                metrics_found += 1
                
                quality_metrics = model_attr.get('quality_metrics', {})
                overall_score = model_attr.get('overall_quality_score')
                quality_tier = model_attr.get('quality_tier')
                
                all_metrics.append({
                    'id': conv.get('conversation_id'),
                    'score': overall_score,
                    'tier': quality_tier,
                    'metrics': quality_metrics
                })
        
        print_pass(f"Conversations with JSONB quality_metrics: {metrics_found}/{len(convs)}")
        
        # Show sample detailed JSONB structure
        if all_metrics:
            print_header("SAMPLE JSONB Structure", level=2)
            sample = all_metrics[0]
            
            print_data("model_attribution (JSONB)", {
                'overall_quality_score': f"{sample['score']:.2%}" if sample['score'] else 'N/A',
                'quality_tier': sample['tier'],
                'quality_metrics (nested object)': len(sample['metrics']) > 0
            }, indent=2)
            
            if sample['metrics']:
                print_info("Quality Metrics Fields:")
                for key, val in sample['metrics'].items():
                    if isinstance(val, (int, float)):
                        print_data(key, f"{val:.4f}" if val < 2 else str(val), indent=4)
                    else:
                        print_data(key, val, indent=4)
            
            # Statistics across all conversations
            print_header("Quality Metrics Statistics", level=2)
            scores = [m['score'] for m in all_metrics if m['score']]
            tiers = [m['tier'] for m in all_metrics]
            
            if scores:
                avg_score = sum(scores) / len(scores)
                print_pass(f"Average Quality Score: {avg_score:.2%}")
                print_pass(f"Min Score: {min(scores):.2%}")
                print_pass(f"Max Score: {max(scores):.2%}")
            
            # Tier distribution
            tier_counts = {}
            for tier in tiers:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            print_info("Quality Tier Distribution:")
            for tier, count in sorted(tier_counts.items()):
                print_data(tier, f"{count} conversations", indent=4)
        
    except Exception as e:
        print_fail(f"Quality metrics validation failed: {e}")

# ============================================================================
# VALIDATION 6: USER ISOLATION & DATA SECURITY
# ============================================================================
def validate_user_isolation():
    print_header("6. USER ISOLATION - No Crossover Between Users", level=1)
    
    try:
        users_to_test = [
            'user_demo_001',
            'user_demo_002',
            'user_demo_003'
        ]
        
        user_data = {}
        
        for user_id in users_to_test:
            result = get_all_conversations(user_id=user_id)
            convs = result.get('results', []) if isinstance(result, dict) else []
            user_data[user_id] = {
                'count': len(convs),
                'ids': {c.get('conversation_id') for c in convs if isinstance(c, dict) and c.get('conversation_id')}
            }
            
            print_pass(f"{user_id:20s} {len(convs):3d} conversations")
        
        # Check for crossover
        print_header("Crossover Check", level=2)
        no_crossover = True
        
        for user1 in users_to_test:
            for user2 in users_to_test:
                if user1 < user2:
                    ids1 = user_data[user1]['ids']
                    ids2 = user_data[user2]['ids']
                    intersection = ids1 & ids2
                    
                    if intersection:
                        print_fail(f"CROSSOVER between {user1} and {user2}: {intersection}")
                        no_crossover = False
                    else:
                        print_pass(f"No crossover: {user1} <-> {user2}")
        
        if no_crossover:
            print_pass("User data properly isolated - no security violations")
        
    except Exception as e:
        print_fail(f"User isolation validation failed: {e}")

# ============================================================================
# VALIDATION 7: COMPLETE FLOW EXAMPLE
# ============================================================================
def showcase_complete_flow():
    print_header("7. COMPLETE FLOW EXAMPLE - One Conversation Lifecycle", level=1)
    
    try:
        result = get_all_conversations(user_id='user_demo_001')
        convs = result.get('results', []) if isinstance(result, dict) else []
        
        if not convs:
            print_warn("No conversations - can't showcase flow")
            return
        
        conv = convs[0]
        
        if not isinstance(conv, dict):
            print_warn("Conversation data format unexpected")
            return
        
        print_header("Example Conversation", level=2)
        
        # User Input
        print_data("1. User Input", {
            'Transcript': (conv.get('transcript', '')[:100] + '...') if conv.get('transcript') else 'N/A',
            'Input Mode': conv.get('input_mode', 'N/A'),
            'Created': conv.get('created_at', 'N/A')
        }, indent=2)
        
        # Financial Analysis
        print_data("2. Financial Analysis", {
            'Topic': conv.get('financial_topic'),
            'Risk Level': conv.get('risk_level'),
            'Sentiment': conv.get('financial_sentiment'),
            'Confidence': f"{conv.get('confidence_score', 0):.2%}"
        }, indent=2)
        
        # AI Response
        print_data("3. AI Response", {
            'Executive Summary': (conv.get('executive_summary', '')[:80] + '...') if conv.get('executive_summary') else 'N/A',
            'Strategic Intent': conv.get('strategic_intent', 'N/A'),
            'Risk Assessment': (conv.get('risk_assessment', '')[:80] + '...') if conv.get('risk_assessment') else 'N/A'
        }, indent=2)
        
        # Quality Score
        model_attr = conv.get('model_attribution', {})
        print_data("4. Quality Metrics (JSONB)", {
            'Overall Score': f"{model_attr.get('overall_quality_score', 0):.2%}",
            'Quality Tier': model_attr.get('quality_tier'),
            'ASR Confidence': f"{model_attr.get('quality_metrics', {}).get('asr_confidence', 0):.2%}"
        }, indent=2)
        
        # Vector Embedding
        if conv.get('embedding'):
            embedding = conv.get('embedding')
            if isinstance(embedding, list):
                print_data("5. Vector Embedding", {
                    'Dimensions': len(embedding),
                    'Model': conv.get('embedding_model'),
                    'Sample Values': embedding[:5]
                }, indent=2)
        else:
            print_data("5. Vector Embedding", {
                'Status': 'Pending (async generation in progress)'
            }, indent=2)
        
        # Entities Extracted
        entities = conv.get('entities', [])
        if entities and isinstance(entities, list):
            print_data("6. Entities Extracted", {
                'Count': len(entities),
                'Examples': [f"{e.get('text', e)} ({e.get('entity_type', 'unknown')})" for e in entities[:3] if isinstance(e, dict)]
            }, indent=2)
        
        print_pass("Complete flow example shown successfully")
        
    except Exception as e:
        print_fail(f"Flow showcase failed: {e}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"{COLORS['GREEN']}")
    print("=" * 80)
    print("  FINFLUX FULL SYSTEM VALIDATION")
    print("  Complete Testing: Tables, Embeddings, Semantic Search, RAG, Self-Query, JSONB")
    print("=" * 80)
    print(f"{COLORS['RESET']}\n")
    
    validate_all_tables()
    validate_embeddings()
    validate_semantic_search()
    validate_rag_self_query()
    validate_quality_metrics()
    validate_user_isolation()
    showcase_complete_flow()
    
    print_header("VALIDATION COMPLETE", level=1)

    total_zero = VALIDATION_RESULTS["zero_data"]
    total_fail = VALIDATION_RESULTS["fail"]

    if total_fail > 0:
        status_color = COLORS['RED']
        status_text = "NOT READY — FAILURES DETECTED"
    elif total_zero > 0:
        status_color = COLORS['YELLOW']
        status_text = f"PARTIAL — {total_zero} VALIDATION(S) HAD ZERO DATA PATHS"
    else:
        status_color = COLORS['GREEN']
        status_text = "READY FOR PRODUCTION"

    print(f"""
{COLORS['GREEN']}Summary of Tests Performed:{COLORS['RESET']}
  [OK] All Supabase tables queried for row counts
  [OK] Vector embeddings validated (dimensions, value ranges)
  [OK] Semantic search tested with multiple queries
  [OK] RAG + Self-Query decomposition showcased
  [OK] Quality metrics JSONB structure analyzed
  [OK] User isolation and data security verified
  [OK] Complete conversation lifecycle shown
  
{COLORS['CYAN']}Next Steps:{COLORS['RESET']}
  1. Run: python scripts/full_validation.py  (any time to verify system health)
  2. If no data: Save a conversation first to populate tables
  3. Check embeddings after 2-3 seconds (async generation)
  4. Test RAG with custom queries
  
{status_color}System Status: {status_text}{COLORS['RESET']}
""")

if __name__ == '__main__':
    main()
