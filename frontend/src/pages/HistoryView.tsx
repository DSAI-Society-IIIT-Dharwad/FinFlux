import { useState, useEffect } from 'react';
import { Clock, ChevronRight, Search, Sparkles, Check, Tag, Globe, AlertCircle } from 'lucide-react';

interface HistoryItem {
  conversation_id: string;
  timestamp: string;
  executive_summary: string;
  strategic_intent: string;
  future_gearing: string;
  risk_assessment: string;
  risk: string;
  risk_level?: string; // Support both
  language: string;
  topic: string;
  financial_topic?: string;
  expert_reasoning?: string;
  similarity_score?: number;
}

export default function HistoryView() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [userId, setUserId] = useState('guest_001'); // Multi-tenant context
  
  // Filters
  const [filterRisk, setFilterRisk] = useState<string | null>(null);
  const [filterLang, setFilterLang] = useState<string | null>(null);
  const [filterTopic, setFilterTopic] = useState<string | null>(null);

  const fetchHistory = (query?: string) => {
    setLoading(true);
    const apiBase = 'http://localhost:8000/api';
    
    if (query || filterRisk || filterLang || filterTopic) {
      // Secure Semantic Search Mode
      fetch(`${apiBase}/search/semantic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          query: query || "Everything",
          filters: {
            risk: filterRisk,
            language: filterLang,
            topic: filterTopic
          }
        })
      })
      .then(res => res.json())
      .then(data => { setItems(data.results || []); setLoading(false); })
      .catch(() => setLoading(false));
    } else {
      // Secure History Mode
      fetch(`${apiBase}/results?user_id=${userId}`)
        .then(res => res.json())
        .then(data => { setItems(data.results || []); setLoading(false); })
        .catch(() => setLoading(false));
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [filterRisk, filterLang, filterTopic, userId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistory(searchQuery);
  };

  const getRiskColor = (l: string) => {
    const risk = l?.toUpperCase();
    if (risk === 'CRITICAL') return '#ef4444';
    if (risk === 'HIGH') return '#f97316';
    if (risk === 'MEDIUM') return '#f59e0b';
    return '#10b981';
  };

  const FilterChip = ({ label, active, onClick, icon: Icon }: any) => (
    <button onClick={onClick} style={{
      display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px',
      borderRadius: '20px', border: '1px solid',
      borderColor: active ? '#3b82f6' : 'rgba(255,255,255,0.1)',
      background: active ? 'rgba(59,130,246,0.1)' : 'transparent',
      color: active ? '#3b82f6' : '#94a3b8',
      fontSize: '0.75rem', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s'
    }}>
      {Icon && <Icon size={12} />}
      {label}
      {active && <Check size={12} />}
    </button>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', color: '#f8fafc', maxWidth: '1000px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ fontSize: '2.25rem', fontWeight: 900, marginBottom: '8px', letterSpacing: '-0.03em' }}>Financial Memory Engine</h2>
          <p style={{ color: '#94a3b8', fontSize: '1rem' }}>AI-Powered Semantic Retrieval & Historical Context</p>
        </div>
        
        {/* Profile Switcher (TESTING ONLY) */}
        <div style={{ display: 'flex', background: 'rgba(255,255,255,0.05)', padding: '4px', borderRadius: '12px' }}>
          {['guest_001', 'guest_002'].map(user => (
            <button key={user} onClick={() => setUserId(user)} style={{
              padding: '6px 12px', borderRadius: '8px', border: 'none',
              background: userId === user ? '#3b82f6' : 'transparent',
              color: userId === user ? 'white' : '#64748b',
              fontSize: '0.7rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s'
            }}>
              {user === 'guest_001' ? 'User Alpha' : 'User Beta'}
            </button>
          ))}
        </div>
      </header>

      {/* Semantic Search Bar */}
      <section className="glass-panel" style={{ padding: '24px', borderRadius: '20px' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <div style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: '#3b82f6' }}>
              <Sparkles size={20} />
            </div>
            <input 
              type="text" 
              placeholder="Search financial memories semantically (e.g., 'worried about emi burden')..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%', padding: '16px 16px 16px 48px', borderRadius: '14px',
                background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.08)',
                color: 'white', fontSize: '1rem', outline: 'none'
              }}
            />
          </div>
          <button type="submit" style={{
            padding: '0 24px', borderRadius: '14px', border: 'none',
            background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
            color: 'white', fontWeight: 600, cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: '8px'
          }}>
            <Search size={18} />
            Search
          </button>
        </form>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center' }}>
          <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 800, textTransform: 'uppercase', marginRight: '4px' }}>Smart Filters :</span>
          
          <FilterChip label="Critical Risk" active={filterRisk === 'CRITICAL'} onClick={() => setFilterRisk(filterRisk === 'CRITICAL' ? null : 'CRITICAL')} icon={AlertCircle} />
          <FilterChip label="Investment" active={filterTopic === 'investment'} onClick={() => setFilterTopic(filterTopic === 'investment' ? null : 'investment')} icon={Tag} />
          <FilterChip label="Loan" active={filterTopic === 'loan'} onClick={() => setFilterTopic(filterTopic === 'loan' ? null : 'loan')} icon={Tag} />
          <FilterChip label="Hindi" active={filterLang === 'Hindi'} onClick={() => setFilterLang(filterLang === 'Hindi' ? null : 'Hindi')} icon={Globe} />
          <FilterChip label="English" active={filterLang === 'English'} onClick={() => setFilterLang(filterLang === 'English' ? null : 'English')} icon={Globe} />
          
          {(filterRisk || filterLang || filterTopic) && (
            <button onClick={() => { setFilterRisk(null); setFilterLang(null); setFilterTopic(null); }}
              style={{ fontSize: '0.7rem', color: '#ef4444', border: 'none', background: 'none', cursor: 'pointer', fontWeight: 700 }}>
              Clear All
            </button>
          )}
        </div>
      </section>

      {loading ? (
        <div style={{ padding: '60px', textAlign: 'center' }}>
          <div className="animate-pulse" style={{ color: '#3b82f6', fontWeight: 700 }}>Activating Semantic Index...</div>
        </div>
      ) : items.length === 0 ? (
        <div className="glass-panel" style={{ padding: '80px 24px', textAlign: 'center' }}>
          <Clock size={48} color="#1e293b" style={{ marginBottom: '16px' }} />
          <h3 style={{ color: '#94a3b8' }}>No Memories Found</h3>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>Try a semantic query or adjust your smart filters.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {items.map((item) => {
            const id = item.conversation_id;
            const isExp = expanded === id;
            const risk = item.risk || item.risk_level || 'LOW';
            
            return (
              <div key={id} className="glass-panel" 
                style={{ 
                  cursor: 'pointer', overflow: 'hidden', 
                  borderLeft: `4px solid ${getRiskColor(risk)}`,
                  transition: 'transform 0.2s'
                }}
                onClick={() => setExpanded(isExp ? null : id)}>
                
                <div style={{ padding: '24px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                      <span style={{ fontSize: '0.65rem', fontWeight: 900, color: 'white', background: getRiskColor(risk), padding: '4px 8px', borderRadius: '4px' }}>{risk} RISK</span>
                      <span style={{ fontSize: '0.65rem', fontWeight: 900, color: '#3b82f6', background: 'rgba(59,130,246,0.1)', padding: '4px 8px', borderRadius: '4px' }}>{(item.topic || item.financial_topic || 'general').toUpperCase()}</span>
                      <span style={{ fontSize: '0.65rem', fontWeight: 900, color: '#10b981', background: 'rgba(16,185,129,0.1)', padding: '4px 8px', borderRadius: '4px' }}>{userId.toUpperCase()}</span>
                      
                      {item.similarity_score !== undefined && (
                        <span style={{ fontSize: '0.65rem', fontWeight: 900, color: '#10b981', background: 'rgba(16,185,129,0.1)', padding: '4px 8px', borderRadius: '4px', marginLeft: 'auto' }}>
                          Similarity: {Math.round(item.similarity_score * 100)}%
                        </span>
                      )}
                    </div>
                    
                    <h4 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '8px', lineHeight: 1.4 }}>{item.executive_summary}</h4>
                    <p style={{ fontSize: '0.8rem', color: '#64748b' }}>{new Date(item.timestamp).toLocaleString()} • {id}</p>
                  </div>
                  <ChevronRight size={20} color="#334155" style={{ marginLeft: '20px', transform: isExp ? 'rotate(90deg)' : 'none', transition: 'transform 0.3s' }} />
                </div>

                {isExp && (
                  <div style={{ padding: '0 24px 24px', animation: 'slideIn 0.3s ease-out' }}>
                    <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', marginBottom: '20px' }} />
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                      <div>
                        <h5 style={{ fontSize: '0.7rem', color: '#3b82f6', textTransform: 'uppercase', fontWeight: 900, marginBottom: '8px' }}>Strategic Intent</h5>
                        <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: 1.6 }}>{item.strategic_intent}</p>
                      </div>
                      <div>
                        <h5 style={{ fontSize: '0.7rem', color: '#10b981', textTransform: 'uppercase', fontWeight: 900, marginBottom: '8px' }}>Future Gearing</h5>
                        <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: 1.6 }}>{item.future_gearing}</p>
                      </div>
                    </div>
                    {item.expert_reasoning && (
                      <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                        <h5 style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 900, marginBottom: '8px' }}>Internal Financial Logic (Qwen)</h5>
                        <p style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.5, whiteSpace: 'pre-line' }}>{item.expert_reasoning}</p>
                      </div>
                    )}
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
