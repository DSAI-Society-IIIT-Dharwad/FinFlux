import { useState } from 'react';
import { Activity, LayoutDashboard, MonitorPlay } from 'lucide-react';
import DashboardView from './pages/DashboardView.tsx';
import RecordView from './pages/RecordView.tsx';

type ViewMode = 'dashboard' | 'record';

function App() {
  const [currentView, setCurrentView] = useState<ViewMode>('dashboard');

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'record', label: 'Analyze Call', icon: MonitorPlay },
  ];

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside style={{
        width: '240px',
        background: 'rgba(13, 15, 20, 0.95)',
        borderRight: '1px solid var(--border-color)',
        padding: '20px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
        backdropFilter: 'blur(20px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #3b82f6, #06b6d4)',
            padding: '8px', borderRadius: '10px',
            boxShadow: '0 0 16px rgba(59,130,246,0.3)',
          }}>
            <Activity color="white" size={18} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.15rem', margin: 0, letterSpacing: '-0.03em' }}>FinFlux</h1>
            <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)', margin: 0, letterSpacing: '0.06em', textTransform: 'uppercase' }}>AI Intelligence</p>
          </div>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map(item => (
            <button key={item.id} onClick={() => setCurrentView(item.id as ViewMode)}
              style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '10px 14px', borderRadius: '10px', border: 'none',
                background: currentView === item.id ? 'rgba(59,130,246,0.12)' : 'transparent',
                color: currentView === item.id ? '#3b82f6' : 'var(--text-secondary)',
                cursor: 'pointer', fontWeight: currentView === item.id ? 600 : 400,
                fontSize: '0.87rem', textAlign: 'left', transition: 'all 0.15s',
              }}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>

        <div style={{ marginTop: 'auto', padding: '12px', borderRadius: '10px', background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.12)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 6px #10b981' }} />
            <span style={{ fontSize: '0.7rem', color: '#10b981', fontWeight: 600 }}>Pipeline Active</span>
          </div>
          <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>FinFlux Whisper-LoRA v1 · Insight Engine</p>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, padding: '32px 40px', overflowY: 'auto', maxHeight: '100vh' }}>
        <div key={currentView} className="animate-fade-in">
          {currentView === 'dashboard' && <DashboardView onStartCapture={() => setCurrentView('record')} />}
          {currentView === 'record' && <RecordView />}
        </div>
      </main>
    </div>
  );
}

export default App;
