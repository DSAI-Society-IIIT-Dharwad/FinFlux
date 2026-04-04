-- Re-enable secure RLS mode
alter table if exists public.conversation_messages enable row level security;
alter table if exists public.message_embeddings enable row level security;
alter table if exists public.conversation_threads enable row level security;
