import { createContext, useContext, useMemo, useState, type ButtonHTMLAttributes, type PropsWithChildren } from 'react';
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';

type SidebarContextValue = {
  collapsed: boolean;
  setCollapsed: (value: boolean) => void;
  toggle: () => void;
};

const SidebarContext = createContext<SidebarContextValue | null>(null);

function cn(...parts: Array<string | undefined | false>) {
  return parts.filter(Boolean).join(' ');
}

export function useSidebar() {
  const value = useContext(SidebarContext);
  if (!value) {
    throw new Error('useSidebar must be used inside SidebarProvider');
  }
  return value;
}

export function SidebarProvider({ children }: PropsWithChildren) {
  const [collapsed, setCollapsed] = useState(false);
  const value = useMemo(
    () => ({
      collapsed,
      setCollapsed,
      toggle: () => setCollapsed((prev) => !prev),
    }),
    [collapsed],
  );

  return (
    <SidebarContext.Provider value={value}>
      <div data-sidebar-state={collapsed ? 'collapsed' : 'expanded'}>{children}</div>
    </SidebarContext.Provider>
  );
}

export function Sidebar({ className, children }: PropsWithChildren<{ className?: string }>) {
  return <aside className={cn('chat-sidebar', className)}>{children}</aside>;
}

export function SidebarHeader({ className, children }: PropsWithChildren<{ className?: string }>) {
  return <div className={cn('sidebar-top', className)}>{children}</div>;
}

export function SidebarContent({ className, children }: PropsWithChildren<{ className?: string }>) {
  return <div className={cn('history-list', className)}>{children}</div>;
}

export function SidebarFooter({ className, children }: PropsWithChildren<{ className?: string }>) {
  return <div className={cn('sidebar-footer', className)}>{children}</div>;
}

export function SidebarGroup({ className, children }: PropsWithChildren<{ className?: string }>) {
  return <div className={className}>{children}</div>;
}

export function SidebarTrigger({ className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  const { collapsed, toggle } = useSidebar();
  return (
    <button
      type="button"
      className={cn('collapse-btn sidebar-trigger', className)}
      onClick={toggle}
      aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      {...props}
    >
      {collapsed ? <PanelLeftOpen size={14} /> : <PanelLeftClose size={14} />}
    </button>
  );
}
