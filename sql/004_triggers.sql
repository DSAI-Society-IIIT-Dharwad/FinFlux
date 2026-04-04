create or replace function public.enforce_message_thread_ownership()
returns trigger
language plpgsql
as $$
declare
  t_owner uuid;
begin
  select user_id into t_owner from public.conversation_threads where id = new.thread_id;
  if t_owner is null then
    raise exception 'Invalid thread_id';
  end if;
  if new.user_id <> t_owner then
    raise exception 'user_id must match thread owner';
  end if;
  return new;
end;
$$;

drop trigger if exists trg_enforce_message_thread_ownership on public.conversation_messages;
create trigger trg_enforce_message_thread_ownership
before insert or update on public.conversation_messages
for each row execute function public.enforce_message_thread_ownership();
