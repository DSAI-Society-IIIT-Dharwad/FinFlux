import { useState, useEffect } from 'react';
import { Mic, Brain, ShieldCheck, Sparkles, ArrowRight, Layers, Zap, Clock, Search, Lock, Loader2, Download } from 'lucide-react';

interface HistoryRecord {
  conversation_id: string;
  timestamp: string;
  financial_topic: string;
  summary: string;
  risk_level: string;
  advice_request: boolean;
  injection_attempt: boolean;
  financial_sentiment: string;
  expert_reasoning: string;
  language: string;
  strategic_intent: string;
  future_gearing: string;
  risk_assessment: string;
}

interface DashboardProps {
  onStartCapture: () => void;
}

export default function DashboardView({ onStartCapture }: DashboardProps) {
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/results')
      .then(r => r.json())
      .then(data => { setHistory(data.results || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const filteredHistory = history.filter((h: HistoryRecord) => 
    (h.summary || "").toLowerCase().includes(search.toLowerCase()) || 
    (h.financial_topic || "").toLowerCase().includes(search.toLowerCase()) ||
    (h.strategic_intent || "").toLowerCase().includes(search.toLowerCase())
  ).sort((a: HistoryRecord, b: HistoryRecord) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  const rc = (l: string) => l === 'CRITICAL' ? '#ef4444' : l === 'HIGH' ? '#f97316' : l === 'MEDIUM' ? '#f59e0b' : '#10b981';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', maxWidth: '1100px', margin: '0 auto', color: '#f8fafc' }}>

      {/* Hero Header V4.2 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '40px', alignItems: 'center', paddingTop: '20px' }}>
        <div>
          <div style={{
            display: 'inline-flex', padding: '6px 16px', borderRadius: '100px',
            background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)',
            marginBottom: '24px', fontSize: '0.75rem', color: '#60a5fa', fontWeight: 700,
            letterSpacing: '0.05em'
          }}>
            <Sparkles size={14} style={{ marginRight: '8px' }} /> FINFLUX PRO V4.2 ENTERPRISE
          </div>
          <h1 style={{
            fontSize: '3.8rem', fontWeight: 900, lineHeight: 1.1,
            background: 'linear-gradient(135deg, #fff 0%, #64748b 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            letterSpacing: '-0.05em', marginBottom: '24px'
          }}>
            Financial Voice<br />Intelligence.
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '1.2rem', maxWidth: '540px', lineHeight: 1.6, marginBottom: '36px' }}>
            A secure 8-stage pipeline for multilingual Indian financial speech. Built for sub-3s extraction with Qwen+FinBERT+GLiNER expert stack.
          </p>
          <div style={{ display: 'flex', gap: '16px' }}>
            <button onClick={onStartCapture}
              style={{ padding: '18px 40px', fontSize: '1.1rem', borderRadius: '18px', background: '#3b82f6', color: 'white', border: 'none', fontWeight: 800, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '12px', boxShadow: '0 10px 25px -10px #3b82f6' }}>
              <Mic size={24} /> Analyze Live <ArrowRight size={20} />
            </button>
          </div>
        </div>

        {/* 8-Stage Pipeline Card */}
        <div style={{
          padding: '32px', borderRadius: '32px',
          background: 'rgba(255,255,255,0.02)',
          border: '1px solid rgba(255,255,255,0.06)',
          display: 'flex', flexDirection: 'column', gap: '22px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ padding: '10px', borderRadius: '12px', background: 'rgba(16,185,129,0.1)', color: '#10b981' }}>
                <ShieldCheck size={24} />
            </div>
            <div>
                <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#10b981' }}>SECURE CORE V4.2</div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>AES-256 + 12 Expert Models</div>
            </div>
          </div>
          <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)' }} />
          {[
            { label: 'ASR Engine', value: 'Whisper Turbo', icon: Zap, color: '#3b82f6' },
            { label: 'Modularity', value: '8-Stage Pipeline', icon: Layers, color: '#f59e0b' },
            { label: 'Expert IQ', value: 'Qwen + Llama 70B', icon: Brain, color: '#a855f7' },
            { label: 'Security', value: 'PII Guarded', icon: Lock, color: '#ef4444' },
          ].map((s, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
               <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <s.icon size={15} color={s.color} />
                  <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{s.label}</span>
               </div>
               <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>{s.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* History Wall */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '1.8rem', fontWeight: 900, letterSpacing: '-0.02em' }}>Intelligence History</h2>
            <div style={{ position: 'relative', width: '300px' }}>
                <Search size={16} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
                <input 
                    placeholder="Search conversations..." 
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    style={{ width: '100%', padding: '12px 16px 12px 42px', borderRadius: '14px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', color: 'white', fontSize: '0.9rem', outline: 'none' }} 
                />
            </div>
        </div>

        {loading ? (
            <div style={{ padding: '80px', textAlign: 'center', opacity: 0.5 }}><Loader2 size={32} className="animate-spin" style={{ margin: '0 auto' }} /></div>
        ) : filteredHistory.length === 0 ? (
            <div style={{ padding: '80px', textAlign: 'center', background: 'rgba(255,255,255,0.01)', borderRadius: '28px', border: '1px dashed rgba(255,255,255,0.1)' }}>
                <Clock size={40} style={{ margin: '0 auto 20px', opacity: 0.1 }} />
                <p style={{ opacity: 0.4 }}>No intelligence logs found.</p>
            </div>
        ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {filteredHistory.map((h, i) => (
                    <div key={i} className="glass-panel" style={{ padding: '24px 32px', display: 'flex', alignItems: 'center', gap: '24px', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '24px', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', width: '130px', flexShrink: 0 }}>
                            <span style={{ fontSize: '0.7rem', opacity: 0.4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Topic IQ</span>
                            <span style={{ fontSize: '1rem', fontWeight: 800, color: '#3b82f6' }}>{h.financial_topic}</span>
                            <div style={{ marginTop: '8px', fontSize: '0.65rem', padding: '4px 8px', background: 'rgba(59,130,246,0.1)', color: '#3b82f6', borderRadius: '6px', textAlign: 'center', border: '1px solid rgba(59,130,246,0.2)' }}>{h.financial_sentiment.toUpperCase()}</div>
                        </div>
                        <div style={{ flex: 1 }}>
                            <p style={{ fontSize: '1.05rem', fontWeight: 500, color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{h.summary}</p>
                            <div style={{ display: 'flex', gap: '16px', marginTop: '10px', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.75rem', opacity: 0.4 }}><Clock size={12} style={{ verticalAlign: 'middle', marginRight: '6px' }} /> {new Date(h.timestamp).toLocaleString()}</span>
                                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                     <span style={{ fontSize: '0.75rem', fontWeight: 800, color: rc(h.risk_level) }}>{h.risk_level} RISK</span>
                                     <div style={{ padding: '4px 8px', borderRadius: '6px', background: 'rgba(16,185,129,0.1)', color: '#10b981', fontSize: '0.65rem', fontWeight: 700 }}>{h.language.toUpperCase()}</div>
                                     {h.strategic_intent && <div style={{ fontSize: '0.65rem', color: '#a855f7', fontWeight: 600, borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '8px' }}>INTENT: {h.strategic_intent}</div>}
                                </div>
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
                            <button 
                              onClick={() => window.open(`http://localhost:8000/api/report/${h.conversation_id}?format=pdf`, '_blank')}
                              title="Download Report" 
                              style={{ width: '40px', height: '40px', borderRadius: '12px', background: 'rgba(255,255,255,0.04)', border: 'none', color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Download size={18} />
                            </button>
                            <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'rgba(59,130,246,0.1)', color: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <ArrowRight size={20} />
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .animate-spin { animation: spin 2s linear infinite; }
        .glass-panel:hover { background: rgba(59,130,246,0.03) !important; border-color: rgba(59,130,246,0.3) !important; transform: translateX(10px); }
      `}</style>
    </div>
  );
}
