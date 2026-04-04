create table if not exists public.users_profile (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.conversation_threads (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text,
  last_message_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.conversation_messages (
  id uuid primary key default gen_random_uuid(),
  thread_id uuid not null references public.conversation_threads(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user','assistant','system')),
  input_mode text not null check (input_mode in ('text','audio')),
  sequence_no bigint not null,
  raw_user_input text,
  transcript text,
  executive_summary text,
  strategic_intent text,
  future_gearing text,
  risk_level text,
  risk_assessment text,
  financial_sentiment text,
  confidence_score numeric,
  model_attribution jsonb not null default '{}'::jsonb,
  expert_reasoning_points text,
  entities jsonb not null default '[]'::jsonb,
  timing jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (thread_id, sequence_no)
);

create table if not exists public.message_embeddings (
  message_id uuid primary key references public.conversation_messages(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  embedding vector(384) not null,
  embedding_model text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.reminders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  thread_id uuid references public.conversation_threads(id) on delete set null,
  message_id uuid references public.conversation_messages(id) on delete set null,
  reminder_text text not null,
  due_at timestamptz,
  status text not null default 'open' check (status in ('open','done')),
  created_at timestamptz not null default now()
);

create table if not exists public.insight_snapshots (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  period_start date not null,
  period_end date not null,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
