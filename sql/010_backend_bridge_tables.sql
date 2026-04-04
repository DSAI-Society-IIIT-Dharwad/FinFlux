-- Backend bridge tables for local JWT users (service-role only access pattern).
-- RLS stays enabled, but AI/API access must go through secure backend endpoints.

create table if not exists public.ai_conversation_threads (
  id text primary key,
  user_id text not null,
  title text,
  last_message_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.ai_conversation_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id text not null,
  thread_id text not null references public.ai_conversation_threads(id) on delete cascade,
  user_id text not null,
  role text not null check (role in ('user','assistant','system')),
  input_mode text not null default 'text' check (input_mode in ('text','audio')),
  sequence_no bigint not null,
  raw_user_input text,
  transcript text,
  executive_summary text,
  strategic_intent text,
  future_gearing text,
  risk_level text,
  risk_assessment text,
  financial_topic text,
  financial_sentiment text,
  confidence_score numeric,
  model_attribution jsonb not null default '{}'::jsonb,
  expert_reasoning_points text,
  entities jsonb not null default '[]'::jsonb,
  timing jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (thread_id, user_id, sequence_no)
);

create table if not exists public.ai_message_embeddings (
  message_id uuid primary key references public.ai_conversation_messages(id) on delete cascade,
  user_id text not null,
  embedding vector(384) not null,
  embedding_model text not null,
  created_at timestamptz not null default now()
);

alter table public.ai_conversation_threads enable row level security;
alter table public.ai_conversation_messages enable row level security;
alter table public.ai_message_embeddings enable row level security;

-- Deny direct reads/writes from anon/authenticated roles.
-- Service role calls from backend are intentionally routed through audited API endpoints.
drop policy if exists ai_threads_no_direct_access on public.ai_conversation_threads;
create policy ai_threads_no_direct_access
on public.ai_conversation_threads
for all
to anon, authenticated
using (false)
with check (false);

drop policy if exists ai_messages_no_direct_access on public.ai_conversation_messages;
create policy ai_messages_no_direct_access
on public.ai_conversation_messages
for all
to anon, authenticated
using (false)
with check (false);

drop policy if exists ai_embeddings_no_direct_access on public.ai_message_embeddings;
create policy ai_embeddings_no_direct_access
on public.ai_message_embeddings
for all
to anon, authenticated
using (false)
with check (false);
