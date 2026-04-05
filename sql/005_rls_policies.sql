alter table public.users_profile enable row level security;
alter table public.conversation_threads enable row level security;
alter table public.conversation_messages enable row level security;
alter table public.message_embeddings enable row level security;
alter table public.reminders enable row level security;
alter table public.insight_snapshots enable row level security;

drop policy if exists users_profile_select_own on public.users_profile;
create policy users_profile_select_own
on public.users_profile for select
to authenticated
using (id = auth.uid());

drop policy if exists users_profile_insert_own on public.users_profile;
create policy users_profile_insert_own
on public.users_profile for insert
to authenticated
with check (id = auth.uid());

drop policy if exists users_profile_update_own on public.users_profile;
create policy users_profile_update_own
on public.users_profile for update
to authenticated
using (id = auth.uid())
with check (id = auth.uid());

drop policy if exists users_profile_delete_own on public.users_profile;
create policy users_profile_delete_own
on public.users_profile for delete
to authenticated
using (id = auth.uid());

drop policy if exists threads_select_own on public.conversation_threads;
create policy threads_select_own
on public.conversation_threads for select
to authenticated
using (user_id = auth.uid());

drop policy if exists threads_insert_own on public.conversation_threads;
create policy threads_insert_own
on public.conversation_threads for insert
to authenticated
with check (user_id = auth.uid());

drop policy if exists threads_update_own on public.conversation_threads;
create policy threads_update_own
on public.conversation_threads for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

drop policy if exists threads_delete_own on public.conversation_threads;
create policy threads_delete_own
on public.conversation_threads for delete
to authenticated
using (user_id = auth.uid());

drop policy if exists messages_select_own on public.conversation_messages;
create policy messages_select_own
on public.conversation_messages for select
to authenticated
using (user_id = auth.uid());

drop policy if exists messages_insert_own on public.conversation_messages;
create policy messages_insert_own
on public.conversation_messages for insert
to authenticated
with check (user_id = auth.uid());

drop policy if exists messages_update_own on public.conversation_messages;
create policy messages_update_own
on public.conversation_messages for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

drop policy if exists messages_delete_own on public.conversation_messages;
create policy messages_delete_own
on public.conversation_messages for delete
to authenticated
using (user_id = auth.uid());

drop policy if exists embeddings_select_own on public.message_embeddings;
create policy embeddings_select_own
on public.message_embeddings for select
to authenticated
using (user_id = auth.uid());

drop policy if exists embeddings_insert_own on public.message_embeddings;
create policy embeddings_insert_own
on public.message_embeddings for insert
to authenticated
with check (user_id = auth.uid());

drop policy if exists embeddings_update_own on public.message_embeddings;
create policy embeddings_update_own
on public.message_embeddings for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

drop policy if exists embeddings_delete_own on public.message_embeddings;
create policy embeddings_delete_own
on public.message_embeddings for delete
to authenticated
using (user_id = auth.uid());

drop policy if exists reminders_select_own on public.reminders;
create policy reminders_select_own
on public.reminders for select
to authenticated
using (user_id = auth.uid());

drop policy if exists reminders_insert_own on public.reminders;
create policy reminders_insert_own
on public.reminders for insert
to authenticated
with check (user_id = auth.uid());

drop policy if exists reminders_update_own on public.reminders;
create policy reminders_update_own
on public.reminders for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

drop policy if exists reminders_delete_own on public.reminders;
create policy reminders_delete_own
on public.reminders for delete
to authenticated
using (user_id = auth.uid());

drop policy if exists insights_select_own on public.insight_snapshots;
create policy insights_select_own
on public.insight_snapshots for select
to authenticated
using (user_id = auth.uid());

drop policy if exists insights_insert_own on public.insight_snapshots;
create policy insights_insert_own
on public.insight_snapshots for insert
to authenticated
with check (user_id = auth.uid());

drop policy if exists insights_update_own on public.insight_snapshots;
create policy insights_update_own
on public.insight_snapshots for update
to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

drop policy if exists insights_delete_own on public.insight_snapshots;
create policy insights_delete_own
on public.insight_snapshots for delete
to authenticated
using (user_id = auth.uid());
