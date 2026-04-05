create index if not exists idx_threads_user_last
  on public.conversation_threads(user_id, last_message_at desc);

create index if not exists idx_messages_user_created
  on public.conversation_messages(user_id, created_at desc);

create index if not exists idx_messages_thread_seq
  on public.conversation_messages(thread_id, sequence_no);

create index if not exists idx_embeddings_user
  on public.message_embeddings(user_id);

create index if not exists idx_embeddings_vec_cos
  on public.message_embeddings
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create index if not exists idx_reminders_user_due
  on public.reminders(user_id, due_at asc);

create index if not exists idx_insights_user_period
  on public.insight_snapshots(user_id, period_start desc);
