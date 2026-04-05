-- Service-key RPC for semantic retrieval on backend bridge tables.
-- User isolation is enforced manually via p_user_id filter in every join.

drop function if exists public.search_user_embeddings_bridge_service(vector, text, integer, text, text, text, double precision);

create or replace function public.search_user_embeddings_bridge_service(
  query_embedding vector(384),
  p_user_id text,
  match_count int default 8,
  filter_thread_id text default null,
  filter_financial_topic text default null,
  filter_risk_level text default null,
  min_similarity double precision default 0.72
)
returns table (
  message_id uuid,
  thread_id text,
  created_at timestamptz,
  similarity float8,
  payload jsonb
)
language sql
stable
security definer
set search_path = public
as $$
  select
    m.id as message_id,
    m.thread_id,
    m.created_at,
    1 - (e.embedding <=> query_embedding) as similarity,
    jsonb_build_object(
      'executive_summary', m.executive_summary,
      'transcript', m.transcript,
      'financial_topic', m.financial_topic,
      'risk_level', m.risk_level,
      'confidence_score', m.confidence_score,
      'financial_sentiment', m.financial_sentiment,
      'strategic_intent', m.strategic_intent,
      'future_gearing', m.future_gearing,
      'risk_assessment', m.risk_assessment
    ) as payload
  from public.ai_message_embeddings e
  join public.ai_conversation_messages m on m.id = e.message_id
  where
    e.user_id = p_user_id
    and m.user_id = p_user_id
    and (filter_thread_id is null or m.thread_id = filter_thread_id)
    and (filter_financial_topic is null or m.financial_topic = filter_financial_topic)
    and (filter_risk_level is null or m.risk_level = filter_risk_level)
    and (1 - (e.embedding <=> query_embedding)) >= min_similarity
  order by e.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

revoke all on function public.search_user_embeddings_bridge_service(vector(384), text, int, text, text, text, double precision) from public;
grant execute on function public.search_user_embeddings_bridge_service(vector(384), text, int, text, text, text, double precision) to service_role;
