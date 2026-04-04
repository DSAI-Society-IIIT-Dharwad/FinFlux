-- Validation and quality metrics for multilingual financial pipeline.
-- Note: ai_conversation_messages.conversation_id is not unique (user + assistant rows),
-- so strict FK-to-conversation_id is emulated via trigger checks and delete cascades.

create table if not exists public.ai_conversation_quality_metrics (
  id uuid primary key default gen_random_uuid(),
  conversation_id text not null,
  user_id text not null,
  asr_confidence numeric,
  ner_coverage_pct numeric,
  rouge1_recall numeric,
  entity_alignment_pct numeric,
  language_confidence numeric,
  financial_relevance_score numeric,
  overall_quality_score numeric,
  quality_tier text check (quality_tier in ('EXCELLENT','GOOD','ACCEPTABLE','LOW')),
  model_versions jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

alter table public.ai_conversation_quality_metrics enable row level security;

-- Service-role-only access pattern (same as 010 backend bridge tables).
drop policy if exists ai_quality_metrics_no_direct_access on public.ai_conversation_quality_metrics;
create policy ai_quality_metrics_no_direct_access
on public.ai_conversation_quality_metrics
for all
to anon, authenticated
using (false)
with check (false);

create index if not exists idx_ai_quality_metrics_user_created_at
  on public.ai_conversation_quality_metrics (user_id, created_at desc);

create index if not exists idx_ai_quality_metrics_conversation_id
  on public.ai_conversation_quality_metrics (conversation_id);

-- Enforce conversation_id existence in ai_conversation_messages.
create or replace function public.enforce_ai_quality_metrics_conversation_exists()
returns trigger
language plpgsql
as $$
begin
  if not exists (
    select 1
    from public.ai_conversation_messages m
    where m.conversation_id = new.conversation_id
  ) then
    raise exception 'conversation_id % not found in ai_conversation_messages', new.conversation_id;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_ai_quality_metrics_conversation_exists on public.ai_conversation_quality_metrics;
create trigger trg_ai_quality_metrics_conversation_exists
before insert or update on public.ai_conversation_quality_metrics
for each row
execute function public.enforce_ai_quality_metrics_conversation_exists();

-- Emulate ON DELETE CASCADE for conversation_id.
create or replace function public.cascade_delete_ai_quality_metrics_by_conversation_id()
returns trigger
language plpgsql
as $$
begin
  delete from public.ai_conversation_quality_metrics q
  where q.conversation_id = old.conversation_id;
  return old;
end;
$$;

drop trigger if exists trg_ai_messages_delete_quality_metrics on public.ai_conversation_messages;
create trigger trg_ai_messages_delete_quality_metrics
after delete on public.ai_conversation_messages
for each row
execute function public.cascade_delete_ai_quality_metrics_by_conversation_id();
