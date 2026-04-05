import { Activity, BarChart3, LogOut, MessageSquare, Plus, Settings, UserRound, Trash2 } from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarHeader,
  useSidebar,
} from './ui/sidebar';

type ViewMode = 'chat' | 'insights' | 'settings';

type ThreadSummary = {
  thread_id: string;
  preview: string;
  topic: string;
  count: number;
};

type AppSidebarProps = {
  view: ViewMode;
  onViewChange: (view: ViewMode) => void;
  onNewChat: () => void;
  threads: ThreadSummary[];
  activeThreadId: string;
  onOpenThread: (threadId: string) => void;
  onDeleteThread: (threadId: string) => void;
  username: string;
  onSignOut: () => void;
};

export function AppSidebar({
  view,
  onViewChange,
  onNewChat,
  threads,
  activeThreadId,
  onOpenThread,
  onDeleteThread,
  username,
  onSignOut,
}: AppSidebarProps) {
  const { collapsed } = useSidebar();

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="brand-row">
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <Activity size={16} /> {!collapsed ? 'FinFlux' : ''}
          </span>
        </div>

        <SidebarGroup>
          <button className="new-chat-btn" onClick={onNewChat}>
            <Plus size={14} /> {!collapsed ? 'New Call Session' : ''}
          </button>
          <button className={`side-tab ${view === 'chat' ? 'active' : ''}`} onClick={() => onViewChange('chat')}>
            <MessageSquare size={14} /> {!collapsed ? 'Call Workspace' : ''}
          </button>
          <button className={`side-tab ${view === 'insights' ? 'active' : ''}`} onClick={() => onViewChange('insights')}>
            <BarChart3 size={14} /> {!collapsed ? 'Call Insights' : ''}
          </button>
          <button className={`side-tab ${view === 'settings' ? 'active' : ''}`} onClick={() => onViewChange('settings')}>
            <Settings size={14} /> {!collapsed ? 'Settings' : ''}
          </button>
        </SidebarGroup>
      </SidebarHeader>

      <SidebarContent>
        {!collapsed && threads.map((item) => (
          <div key={item.thread_id} className="history-item-container">
            <button
              className={`history-item ${activeThreadId === item.thread_id ? 'active' : ''}`}
              onClick={() => onOpenThread(item.thread_id)}
            >
              <div className="history-topic">Call Log · {item.topic || 'General'} · {item.count}</div>
              {!collapsed && <div className="history-snippet">{(item.preview || '').slice(0, 70)}</div>}
            </button>
            {!collapsed && (
              <button
                className="history-delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteThread(item.thread_id);
                }}
                title="Delete this chat"
              >
                <Trash2 size={14} />
              </button>
            )}
          </div>
        ))}
      </SidebarContent>

      <SidebarFooter>
        <div className="user-chip">
          <UserRound size={14} /> {!collapsed ? username : ''}
        </div>
        <button className="signout-btn" onClick={onSignOut}>
          <LogOut size={14} /> {!collapsed ? 'Logout' : ''}
        </button>
      </SidebarFooter>
    </Sidebar>
  );
}
