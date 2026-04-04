import { useState, useEffect } from 'react';
import { Clock, ChevronRight } from 'lucide-react';

interface HistoryItem {
  conversation_id: string;
  call_id?: string; // Legacy support
  timestamp: string;
  is_financial: boolean;
  summary: string;
  executive_summary?: string;
  strategic_intent?: string;
  future_gearing?: string;
  risk_assessment?: string;
  commitments: number;
  entities: any[];
  risk_level: string;
  language: string;
  detected_language?: string; // Legacy support
  financial_topic: string;
  detected_topics?: string[]; // Legacy support
  expert_reasoning: string;
}

export default function HistoryView() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/results')
      .then(res => res.json())
      .then(data => { setItems((data.results || [])); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const rc = (l: string) => l === 'CRITICAL' ? '#ef4444' : l === 'HIGH' ? '#f97316' : l === 'MEDIUM' ? '#f59e0b' : '#10b981';
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', color: '#f8fafc' }}>
      <div>
        <h2 style={{ fontSize: '2rem', fontWeight: 900, marginBottom: '4px' }}>Strategic Intelligence History</h2>
        <p style={{ color: '#94a3b8' }}>V4.2+ McKinsey-Level Discovery Logs</p>
      </div>

      {loading ? (
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>Executing Database Query...</div>
      ) : items.length === 0 ? (
        <div className="glass-panel" style={{ padding: '56px 24px', textAlign: 'center' }}>
          <Clock size={40} color="#94a3b8" style={{ marginBottom: '12px', opacity: 0.2 }} />
          <h3 style={{ color: '#94a3b8', marginBottom: '6px' }}>No Strategic Logs</h3>
          <p style={{ color: '#64748b', maxWidth: '380px', margin: '0 auto', fontSize: '0.9rem' }}>
            Capture new financial intelligence to populate this wall.
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {items.map((item) => {
            const id = item.conversation_id || item.call_id || 'unknown';
            const isExp = expanded === id;
            return (
              <div key={id} className="glass-panel" 
                style={{ cursor: 'pointer', overflow: 'hidden', background: 'rgba(255,255,255,0.015)', transition: 'all 0.3s' }}
                onClick={() => setExpanded(isExp ? null : id)}>
                
                <div style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '18px' }}>
                    <div style={{ width: '4px', height: '24px', borderRadius: '2px', background: rc(item.risk_level) }} />
                    <div>
                      <p style={{ fontWeight: 800, fontSize: '0.95rem', marginBottom: '2px', color: '#3b82f6' }}>{item.financial_topic}</p>
                      <p style={{ fontSize: '0.75rem', color: '#64748b' }}>{new Date(item.timestamp).toLocaleString()} • {id}</p>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <span style={{ fontSize: '0.7rem', fontWeight: 900, color: '#10b981', background: 'rgba(16,185,129,0.1)', padding: '4px 10px', borderRadius: '6px' }}>{(item.language || item.detected_language || 'en').toUpperCase()}</span>
                    <span style={{ fontSize: '0.7rem', fontWeight: 900, color: rc(item.risk_level), background: `${rc(item.risk_level)}15`, padding: '4px 10px', borderRadius: '6px' }}>{item.risk_level} RISK</span>
                    <ChevronRight size={18} color="#64748b" style={{ transform: isExp ? 'rotate(90deg)' : 'none', transition: 'transform 0.3s' }} />
                  </div>
                </div>

                {isExp && (
                  <div style={{ padding: '0 24px 24px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '20px' }} className="animate-in fade-in slide-in-from-top-4">
                    <p style={{ fontSize: '1rem', color: '#f8fafc', lineHeight: 1.6, marginBottom: '20px', fontWeight: 500 }}>{item.executive_summary || item.summary}</p>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
                        <div style={{ padding: '16px', background: 'rgba(59,130,246,0.04)', borderRadius: '12px', border: '1px solid rgba(59,130,246,0.1)' }}>
                            <h5 style={{ fontSize: '0.65rem', color: '#3b82f6', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 900 }}>Strategic Intent</h5>
                            <p style={{ fontSize: '0.85rem', opacity: 0.8 }}>{item.strategic_intent || 'N/A'}</p>
                        </div>
                        <div style={{ padding: '16px', background: 'rgba(168,85,247,0.04)', borderRadius: '12px', border: '1px solid rgba(168,85,247,0.1)' }}>
                            <h5 style={{ fontSize: '0.65rem', color: '#a855f7', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 900 }}>Future Gearing</h5>
                            <p style={{ fontSize: '0.85rem', opacity: 0.8 }}>{item.future_gearing || 'N/A'}</p>
                        </div>
                        <div style={{ padding: '16px', background: 'rgba(239,68,68,0.04)', borderRadius: '12px', border: '1px solid rgba(239,68,68,0.1)' }}>
                            <h5 style={{ fontSize: '0.65rem', color: '#ef4444', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 900 }}>Risk Assessment</h5>
                            <p style={{ fontSize: '0.85rem', opacity: 0.8 }}>{item.risk_assessment || 'N/A'}</p>
                        </div>
                    </div>

                    <div style={{ marginTop: '20px' }}>
                        <h5 style={{ fontSize: '0.65rem', color: '#64748b', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 900 }}>Expert Reasoning (Qwen)</h5>
                        <p style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.5, whiteSpace: 'pre-line' }}>{item.expert_reasoning}</p>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
