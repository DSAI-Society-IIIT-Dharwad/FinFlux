drop function if exists public.search_user_message_embeddings(vector, integer, uuid, double precision);

create or replace function public.search_user_message_embeddings(
  query_embedding vector(384),
  match_count int default 8,
  filter_thread_id uuid default null,
  min_similarity double precision default 0.72
)
returns table (
  message_id uuid,
  thread_id uuid,
  created_at timestamptz,
  similarity float8,
  payload jsonb
)
language sql
stable
security invoker
as $$
  select
    m.id as message_id,
    m.thread_id,
    m.created_at,
    1 - (e.embedding <=> query_embedding) as similarity,
    jsonb_build_object(
      'executive_summary', to_jsonb(m)->>'executive_summary',
      'transcript', to_jsonb(m)->>'transcript',
      'financial_topic', coalesce(to_jsonb(m)->>'financial_topic', to_jsonb(m)->>'topic'),
      'risk_level', to_jsonb(m)->>'risk_level',
      'confidence_score', to_jsonb(m)->>'confidence_score',
      'financial_sentiment', to_jsonb(m)->>'financial_sentiment',
      'strategic_intent', to_jsonb(m)->>'strategic_intent',
      'future_gearing', to_jsonb(m)->>'future_gearing',
      'risk_assessment', to_jsonb(m)->>'risk_assessment'
    ) as payload
  from public.message_embeddings e
  join public.conversation_messages m on m.id = e.message_id
  where
    e.user_id = auth.uid()
    and m.user_id = auth.uid()
    and (filter_thread_id is null or m.thread_id = filter_thread_id)
    and (1 - (e.embedding <=> query_embedding)) >= min_similarity
  order by e.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

grant execute on function public.search_user_message_embeddings(vector(384), int, uuid, double precision) to authenticated;
