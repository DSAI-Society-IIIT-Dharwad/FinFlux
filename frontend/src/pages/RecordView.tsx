import { useState, useRef, useCallback } from 'react';
import { Mic, Square, Loader2, Tag, ShieldAlert, Sparkles, TrendingUp, ShieldCheck, Download, Edit2, Save, FileText, CheckCircle } from 'lucide-react';

type SpeechRecognitionCtor = new () => {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  start: () => void;
  stop: () => void;
};

interface Entity { type: string; value: string; context: string; confidence?: number }
interface TopicScore { topic: string; score: number }
interface SentimentBreakdown { positive?: number; neutral?: number; negative?: number }
interface ModelAttribution {
  xlm_roberta?: { detected_language?: string; confidence?: number };
  deberta?: { top_topic?: string; top3_topics?: TopicScore[] };
  finbert?: { label?: string; breakdown?: SentimentBreakdown };
  gliner?: { entity_count?: number };
  qwen?: { reasoning_available?: boolean; section?: string };
}
interface TimingData { total_s?: number; asr_s?: number; normalization_s?: number; expert_nlp_s?: number; synthesis_s?: number }
interface RagRetrievedRow {
  conversation_id?: string;
  score?: number;
  document?: string;
  metadata?: {
    timestamp?: string;
    risk_level?: string;
    topic?: string;
    sentiment?: string;
    language?: string;
  };
}
interface RagInsight {
  contextual_insight?: string;
  pattern_summary?: string[];
  risk_trajectory?: string;
  recommended_next_focus?: string[];
  confidence?: string;
}
interface RagResponse {
  query?: string;
  retrieval?: { available?: boolean; plan?: { semantic_query?: string; filters?: Record<string, unknown>; mode?: string; top_k?: number }; results?: RagRetrievedRow[] };
  rag_context?: string;
  insight?: RagInsight;
}
interface AnalysisResult {
  conversation_id: string; timestamp: string; language: string;
  financial_topic: string; risk_level: string;
  advice_request: boolean; injection_attempt: boolean;
  entities: Entity[]; executive_summary: string; key_insights: string[];
  confidence_score: number; timing: TimingData;
  financial_sentiment: string;
  expert_reasoning_points: string;
  future_gearing: string;
  strategic_intent: string;
  risk_assessment: string;
  transcript: string;
  language_confidence?: number;
  topic_top3?: TopicScore[];
  sentiment_breakdown?: SentimentBreakdown;
  model_attribution?: ModelAttribution;
  financial_terms?: string[];
  financial_parameters?: Record<string, string[]>;
  financial_nlp_topic?: string;
  financial_nlp_topic_confidence?: number;
  financial_nlp?: Record<string, unknown>;
}

function encodeWavBlob(samples: Float32Array, sr: number): Blob {
  const bps = 16, nc = 1, ba = nc * (bps / 8), dl = samples.length * ba;
  const buf = new ArrayBuffer(44 + dl), v = new DataView(buf);
  const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0,'RIFF'); v.setUint32(4,36+dl,true); ws(8,'WAVE'); ws(12,'fmt ');
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,nc,true);
  v.setUint32(24,sr,true); v.setUint32(28,sr*ba,true); v.setUint16(32,ba,true);
  v.setUint16(34,bps,true); ws(36,'data'); v.setUint32(40,dl,true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

export default function RecordView() {
  const [mode, setMode] = useState<'idle' | 'recording' | 'processing' | 'done'>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [ragResult, setRagResult] = useState<RagResponse | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [financialDetected, setFinancialDetected] = useState(false);
  const [matchedKeywords, setMatchedKeywords] = useState<string[]>([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchRagInsight = async (analysis: AnalysisResult) => {
    setRagLoading(true);
    setRagError('');
    try {
      const querySource = (analysis.executive_summary || analysis.transcript || '').trim();
      const response = await fetch('http://localhost:8000/api/rag/insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: querySource || 'Summarize relevant historical financial patterns.',
          top_k: 6,
          financial_sentiment: analysis.financial_sentiment || 'Neutral',
          filters: {
            language: (analysis.language || '').toLowerCase() || 'unknown',
            risk_level: analysis.risk_level || undefined,
            topic: analysis.financial_topic || undefined,
          },
        }),
      });

      if (!response.ok) {
        let message = `RAG unavailable (${response.status})`;
        try {
          const body = await response.json();
          if (body?.detail) message = String(body.detail);
        } catch {
          // keep fallback message
        }
        setRagResult(null);
        setRagError(message);
        return;
      }

      const data = await response.json();
      setRagResult(data);
    } catch {
      setRagResult(null);
      setRagError('Could not fetch RAG output');
    } finally {
      setRagLoading(false);
    }
  };

  const saveChanges = async () => {
    if(!result) return;
    setIsSaving(true);
    try {
      const res = await fetch(`http://localhost:8000/api/update/${result.conversation_id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: result.transcript, executive_summary: result.executive_summary })
      });
      if(res.ok) {
        setIsEditing(false);
        void fetchRagInsight(result);
      }
      else { setErrorMsg('Save failed'); }
    } catch { setErrorMsg('Connection failed'); }
    finally { setIsSaving(false); }
  };
  const audioCtxRef = useRef<AudioContext | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const pcmBufferRef = useRef<Float32Array[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);
  const chunkDetectTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const chunkBufferRef = useRef<string>('');

  const drawWaveform = useCallback(() => {
    const c = canvasRef.current, a = analyserRef.current;
    if (!c || !a) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    const len = a.frequencyBinCount, d = new Uint8Array(len);
    a.getByteTimeDomainData(d);
    ctx.fillStyle = 'black'; ctx.fillRect(0,0,c.width,c.height);

    ctx.lineWidth = 3; ctx.strokeStyle = '#3b82f6'; ctx.shadowColor = '#3b82f6'; ctx.shadowBlur = 12;
    ctx.beginPath();
    const sw = c.width / len; let x = 0;
    for (let i = 0; i < len; i++) {
      const y = (d[i]/128)*(c.height/2);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
      x+=sw;
    }
    ctx.lineTo(c.width, c.height/2); ctx.stroke(); ctx.shadowBlur = 0;
    animFrameRef.current = requestAnimationFrame(drawWaveform);
  }, []);

  const startRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const actx = new AudioContext({ sampleRate: 16000 }); audioCtxRef.current = actx;
      const src = actx.createMediaStreamSource(stream);
      const an = actx.createAnalyser(); an.fftSize = 2048; src.connect(an); analyserRef.current = an;
      pcmBufferRef.current = [];
      const sn = actx.createScriptProcessor(4096, 1, 1);
      sn.onaudioprocess = (e) => pcmBufferRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(sn); sn.connect(actx.destination); scriptNodeRef.current = sn;
      setMode('recording'); setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime(t=>t+1), 1000);
      drawWaveform();

      setFinancialDetected(false);
      setMatchedKeywords([]);
      chunkBufferRef.current = '';

      const SR = ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition) as SpeechRecognitionCtor | undefined;
      if (SR) {
        const rec = new SR();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = 'en-IN';
        rec.onresult = (event: any) => {
          let chunk = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            chunk += event.results[i][0].transcript || '';
          }
          if (chunk.trim()) {
            chunkBufferRef.current = `${chunkBufferRef.current} ${chunk}`.trim();
          }
        };
        rec.onerror = () => {
          // Continue recording even if speech recognition fails.
        };
        recognitionRef.current = rec;
        try { rec.start(); } catch {}
      }

      chunkDetectTimerRef.current = setInterval(async () => {
        const textChunk = chunkBufferRef.current.trim();
        chunkBufferRef.current = '';
        if (!textChunk) return;
        try {
          const res = await fetch('http://localhost:8000/api/realtime/financial-detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textChunk }),
          });
          if (!res.ok) return;
          const data = await res.json();
          const detected = Boolean(data?.financial_detected);
          const words = Array.isArray(data?.matched_keywords) ? data.matched_keywords : [];
          setFinancialDetected(detected);
          setMatchedKeywords(words.slice(0, 6));
        } catch {
          // Ignore transient errors for realtime indicator path.
        }
      }, 5000);
    } catch { setErrorMsg('Mic Error'); }
  };

  const stopRecord = async () => {
    cancelAnimationFrame(animFrameRef.current);
    if(timerRef.current) clearInterval(timerRef.current);
    if(chunkDetectTimerRef.current) clearInterval(chunkDetectTimerRef.current);
    if(recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch {}
      recognitionRef.current = null;
    }
    streamRef.current?.getTracks().forEach(t=>t.stop());
    scriptNodeRef.current?.disconnect();
    
    const chunks = pcmBufferRef.current, total = chunks.reduce((s,c)=>s+c.length,0);
    const merged = new Float32Array(total); let off=0;
    for(const c of chunks){ merged.set(c,off); off+=c.length; }
    const wav = new File([encodeWavBlob(merged, 16000)], 'rec.wav', { type: 'audio/wav' });
    setMode('processing');
    
    const fd = new FormData(); 
    fd.append('file', wav);
    fd.append('user_id', 'guest_001'); // Mandatory tenant tag
    setRagResult(null);
    setRagError('');

    try {
        const res = await fetch('http://localhost:8000/api/analyze', { method: 'POST', body: fd });
      if(res.ok) {
        const analysis = await res.json();
        setResult(analysis);
        setMode('done');
        void fetchRagInsight(analysis);
      }
        else { setMode('idle'); setErrorMsg('Server Error: ' + res.statusText); }
    } catch (e) {
        setMode('idle'); setErrorMsg('Connection Error: ' + e);
    }
  };

  const downloadReport = (type: 'pdf' | 'csv') => {
    if(!result) return;
    window.open(`http://localhost:8000/api/report/${result.conversation_id}?format=${type}`, '_blank');
  };

  const rc = (l: string) => l === 'CRITICAL' ? '#ef4444' : l === 'HIGH' ? '#f97316' : l === 'MEDIUM' ? '#f59e0b' : '#10b981';
  const pct = (value?: number) => `${Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)))}%`;
  const safeTopics = (result?.model_attribution?.deberta?.top3_topics || result?.topic_top3 || []);
  const sentiment = result?.model_attribution?.finbert?.breakdown || result?.sentiment_breakdown || {};
  const financialTerms = result?.financial_terms || [];
  const financialParameters = result?.financial_parameters || {};
  const ragRows = ragResult?.retrieval?.results || [];
  const ragInsight = ragResult?.insight;
  const extractSummary = (doc?: string) => {
    if (!doc) return 'No summary available.';
    if (doc.includes('Executive Summary:')) {
      return doc.split('Executive Summary:')[1]?.split('Transcript:')[0]?.trim() || 'No summary available.';
    }
    return doc.slice(0, 220);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', maxWidth: '1200px', margin: '0 auto', color: '#f8fafc' }}>
      
      {/* McKinsey Gating Header */}
      <div style={{ textAlign: 'center', padding: '40px 0 20px' }}>
        <h1 style={{ fontSize: '2.8rem', fontWeight: 900, letterSpacing: '-0.05em', background: 'linear-gradient(135deg, #f8fafc 0%, #3b82f6 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          {mode === 'idle' ? 'Secure IQ Gating' : mode === 'recording' ? 'Strategic Capture' : mode === 'processing' ? 'McKinsey IQ Pipeline' : 'Strategic Intelligence Wall'}
        </h1>
        <p style={{ color: '#94a3b8', marginTop: '10px', fontSize: '1rem', fontWeight: 600 }}>MCKINSEY-LEVEL STRATEGIC INSIGHTS · 12-MODEL PRO PIPELINE · V4.2+</p>
      </div>

      {errorMsg && (
        <div className="animate-in fade-in" style={{ padding: '16px 24px', borderRadius: '16px', background: 'rgba(239,68,68,0.1)', border: '1px solid #ef4444', color: '#ef4444', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <ShieldAlert size={20} />
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{errorMsg}</span>
        </div>
      )}

      {/* Capture Area */}
      {mode !== 'done' && (
        <div style={{ padding: '100px 60px', borderRadius: '40px', background: 'rgba(255,255,255,0.015)', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
          {mode === 'recording' && <canvas ref={canvasRef} width={800} height={120} style={{ position: 'absolute', bottom: 0, left: 0, width: '100%', opacity: 0.2 }} />}

          {mode === 'recording' && (
            <div style={{ position: 'absolute', top: '16px', right: '16px', display: 'flex', gap: '8px', alignItems: 'center' }}>
              <div style={{ padding: '8px 12px', borderRadius: '999px', border: `1px solid ${financialDetected ? 'rgba(16,185,129,0.45)' : 'rgba(148,163,184,0.35)'}`, background: financialDetected ? 'rgba(16,185,129,0.15)' : 'rgba(148,163,184,0.1)', color: financialDetected ? '#10b981' : '#94a3b8', fontSize: '0.72rem', fontWeight: 800, letterSpacing: '0.04em' }}>
                {financialDetected ? 'FINANCIAL DETECTED' : 'LISTENING'}
              </div>
              {matchedKeywords.length > 0 && (
                <div style={{ padding: '8px 12px', borderRadius: '999px', border: '1px solid rgba(59,130,246,0.35)', background: 'rgba(59,130,246,0.15)', color: '#93c5fd', fontSize: '0.72rem', fontWeight: 700, maxWidth: '340px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {matchedKeywords.join(', ')}
                </div>
              )}
            </div>
          )}
          
          <button onClick={mode === 'idle' ? startRecord : stopRecord} 
            style={{ width: '140px', height: '140px', borderRadius: '50%', border: 'none', background: mode === 'recording' ? '#ef4444' : '#3b82f6', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto', transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)', transform: mode === 'recording' ? 'scale(1.15)' : 'scale(1)', boxShadow: mode === 'recording' ? '0 0 80px rgba(239,68,68,0.5)' : '0 0 80px rgba(59,130,246,0.2)' }}>
            {mode === 'idle' ? <Mic size={64} color="white" /> : <Square size={52} color="white" />}
          </button>
          
          <div style={{ marginTop: '50px' }}>
            {mode === 'recording' ? <span style={{ fontSize: '2.2rem', fontFamily: 'monospace', fontWeight: 900, color: 'white' }}>{Math.floor(recordingTime/60)}:{(recordingTime%60).toString().padStart(2,'0')}</span> :
             mode === 'processing' ? <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'center' }}><Loader2 className="spin" size={40} color="#3b82f6" /><span style={{ fontSize: '1.2rem', opacity: 0.8, fontWeight: 700, letterSpacing: '0.05em' }}>Executing McKinsey Synthesis Engine (Llama 70B)...</span></div> :
             <span style={{ fontSize: '1.1rem', opacity: 0.4, fontWeight: 600 }}>Tap to Capture High-Value Financial Intel</span>}
          </div>
        </div>
      )}

      {/* McKinsey Strategic Wall (V4.2+) */}
      {result && mode === 'done' && (
        <div className="animate-in fade-in slide-in-from-bottom-20 duration-1000" style={{ display: 'grid', gridTemplateColumns: '7.5fr 4.5fr', gap: '32px' }}>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Top Strategic Toolbar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '12px' }}>
                     <div style={{ padding: '8px 16px', background: 'rgba(59,130,246,0.1)', borderRadius: '100px', fontSize: '0.8rem', fontWeight: 900, color: '#3b82f6', border: '1px solid rgba(59,130,246,0.2)', letterSpacing: '0.05em' }}>MCKINSEY ANALYSIS</div>
                     <div style={{ padding: '8px 16px', background: 'rgba(16,185,129,0.1)', borderRadius: '100px', fontSize: '0.8rem', fontWeight: 900, color: '#10b981', border: '1px solid rgba(16,185,129,0.2)', letterSpacing: '0.05em' }}>{result.financial_sentiment.toUpperCase()} SENTIMENT</div>
                </div>
                <div style={{ display: 'flex', gap: '15px' }}>
                    <button onClick={()=>downloadReport('pdf')} className="btn-m" style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(255,255,255,0.03)', padding: '10px 18px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', cursor: 'pointer', color: 'white', fontWeight: 700 }}><Download size={16} /> PDF Report</button>
                    {isEditing ? (
                      <button onClick={saveChanges} style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#10b981', padding: '10px 18px', borderRadius: '12px', border: 'none', cursor: 'pointer', color: 'white', fontWeight: 700 }}>
                        {isSaving ? <Loader2 className="spin" size={16} /> : <Save size={16} />} Save Changes
                      </button>
                    ) : (
                      <button onClick={()=>setIsEditing(true)} style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 18px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', cursor: 'pointer', color: 'white', fontWeight: 700 }}><Edit2 size={16} /></button>
                    )}
                </div>
            </div>

            {/* Strategic Summary Panel */}
            <div className="glass-panel" style={{ padding: '40px', background: 'rgba(255,255,255,0.015)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                <h3 style={{ margin: 0, color: '#3b82f6', display: 'flex', alignItems: 'center', gap: '12px', fontSize: '1.5rem', fontWeight: 900 }}><Sparkles size={24} /> Executive Strategic Summary</h3>
              </div>
              
              {isEditing ? (
                  <textarea value={result.executive_summary} onChange={(e)=>setResult({...result, executive_summary: e.target.value})} style={{ width: '100%', height: '140px', background: 'rgba(0,0,0,0.4)', border: '1px solid #3b82f6', borderRadius: '16px', color: 'white', padding: '20px', fontSize: '1.1rem', outline: 'none' }} />
              ) : (
                  <p style={{ fontSize: '1.3rem', lineHeight: 1.7, fontWeight: 500, color: '#f8fafc' }}>{result.executive_summary}</p>
              )}

              {/* Verified Transcript Section */}
              <div style={{ marginTop: '32px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '32px' }}>
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1rem', color: '#64748b', marginBottom: '16px', fontWeight: 800 }}><FileText size={18} /> Verified Transcript</h4>
                {isEditing ? (
                  <textarea value={result.transcript} onChange={(e)=>setResult({...result, transcript: e.target.value})} style={{ width: '100%', height: '200px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', color: '#94a3b8', padding: '20px', fontSize: '1rem', outline: 'none', lineHeight: 1.6 }} />
                ) : (
                  <div style={{ marginTop: '16px' }}>
                      {result.transcript
                        .split('\n\n')
                        .filter(p => p.trim())
                        .map((paragraph, i) => (
                          <p key={i} style={{ 
                            fontSize: '1.02rem', 
                            color: '#94a3b8', 
                            lineHeight: 2.0, 
                            marginBottom: '20px',
                            borderLeft: paragraph.trim().startsWith('Speaker') ? '3px solid #3b82f6' : 'none',
                            paddingLeft: paragraph.trim().startsWith('Speaker') ? '16px' : '0'
                          }}>
                            {paragraph}
                          </p>
                        ))
                      }
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '32px' }}>
                  <div style={{ padding: '24px', background: 'rgba(59,130,246,0.03)', borderRadius: '20px', border: '1px solid rgba(59,130,246,0.2)' }}>
                      <h4 style={{ fontSize: '0.8rem', color: '#3b82f6', marginBottom: '12px', textTransform: 'uppercase', fontWeight: 900, letterSpacing: '0.05em' }}>Strategic Intent</h4>
                      <p style={{ fontSize: '1rem', opacity: 0.9, lineHeight: 1.5 }}>{result.strategic_intent}</p>
                  </div>
                  <div style={{ padding: '24px', background: 'rgba(168,85,247,0.03)', borderRadius: '20px', border: '1px solid rgba(168,85,247,0.2)' }}>
                      <h4 style={{ fontSize: '0.8rem', color: '#a855f7', marginBottom: '12px', textTransform: 'uppercase', fontWeight: 900, letterSpacing: '0.05em' }}>Future Gearing</h4>
                      <p style={{ fontSize: '1rem', opacity: 0.9, lineHeight: 1.5 }}>{result.future_gearing}</p>
                  </div>
              </div>

              {/* Wall of Logic (Qwen) */}
              <div style={{ marginTop: '40px', padding: '32px', background: 'linear-gradient(180deg, rgba(16,185,129,0.08) 0%, rgba(0,0,0,0.35) 100%)', borderRadius: '24px', border: '1px solid rgba(16,185,129,0.25)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '0.9rem', color: '#10b981', textTransform: 'uppercase', fontWeight: 900, letterSpacing: '0.1em', display: 'flex', alignItems: 'center', gap: '12px' }}><CheckCircle size={18} color="#10b981" /> Wall of Logic (Qwen)</h4>
                  <span style={{ fontSize: '0.72rem', fontWeight: 800, color: '#86efac', letterSpacing: '0.08em' }}>{result.model_attribution?.qwen?.reasoning_available ? 'ACTIVE' : 'DEGRADED'}</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {result.expert_reasoning_points
                    .split('\n')
                    .filter(line => line.trim())
                    .map((line, i) => {
                      const parts = line.split(/\*\*(.*?)\*\*/g);
                      return (
                        <div key={i} style={{ border: '1px solid rgba(255,255,255,0.05)', borderRadius: '14px', padding: '14px 16px', background: 'rgba(0,0,0,0.25)' }}>
                          <p style={{ fontSize: '1rem', color: line.trim().startsWith('•') ? '#f1f5f9' : '#cbd5e1', lineHeight: 1.8, margin: 0 }}>
                            {parts.map((part, j) =>
                              j % 2 === 1
                                ? <strong key={j} style={{ color: '#34d399', fontWeight: 800 }}>{part}</strong>
                                : part
                            )}
                          </p>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>

            {/* Strategic Topic Card */}
            <div className="glass-panel" style={{ padding: '32px' }}>
               <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h4 style={{ fontSize: '0.8rem', marginBottom: '8px', opacity: 0.4, textTransform: 'uppercase', fontWeight: 900 }}>Strategic Gating</h4>
                    <div style={{ fontSize: '1.8rem', fontWeight: 900, letterSpacing: '-0.02em' }}>{result.financial_topic}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#3b82f6' }}>{Math.round(result.confidence_score*100)}%</div>
                    <div style={{ fontSize: '0.7rem', opacity: 0.4 }}>CONFIDENCE IQ</div>
                  </div>
               </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {/* Model Attribution */}
            <div className="glass-panel" style={{ padding: '28px', background: 'rgba(255,255,255,0.02)' }}>
              <h4 style={{ marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1rem', fontWeight: 900, color: '#38bdf8' }}>
                <Sparkles size={18} /> Model Attribution
              </h4>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                <div style={{ padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.25)' }}>
                  <div style={{ fontSize: '0.75rem', color: '#93c5fd', fontWeight: 800, letterSpacing: '0.07em' }}>XLM-ROBERTA · LANGUAGE DETECTION</div>
                  <div style={{ marginTop: '8px', fontWeight: 700 }}>{result.model_attribution?.xlm_roberta?.detected_language || result.language}</div>
                  <div style={{ marginTop: '8px', height: '8px', borderRadius: '999px', background: 'rgba(255,255,255,0.08)' }}>
                    <div style={{ width: pct(result.model_attribution?.xlm_roberta?.confidence ?? result.language_confidence), height: '100%', borderRadius: '999px', background: 'linear-gradient(90deg, #38bdf8, #3b82f6)' }} />
                  </div>
                  <div style={{ marginTop: '6px', fontSize: '0.75rem', color: '#93c5fd' }}>{pct(result.model_attribution?.xlm_roberta?.confidence ?? result.language_confidence)} confidence</div>
                </div>

                <div style={{ padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.25)' }}>
                  <div style={{ fontSize: '0.75rem', color: '#fda4af', fontWeight: 800, letterSpacing: '0.07em' }}>DEBERTA · TOPIC CLASSIFICATION</div>
                  <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {safeTopics.length === 0 ? (
                      <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>No topic probabilities available.</div>
                    ) : safeTopics.map((t, i) => (
                      <div key={`${t.topic}-${i}`}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                          <span style={{ color: '#f1f5f9', textTransform: 'capitalize' }}>{t.topic}</span>
                          <span style={{ color: '#fda4af', fontWeight: 800 }}>{pct(t.score)}</span>
                        </div>
                        <div style={{ marginTop: '4px', height: '6px', borderRadius: '999px', background: 'rgba(255,255,255,0.08)' }}>
                          <div style={{ width: pct(t.score), height: '100%', borderRadius: '999px', background: 'linear-gradient(90deg, #fb7185, #f43f5e)' }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.25)' }}>
                  <div style={{ fontSize: '0.75rem', color: '#86efac', fontWeight: 800, letterSpacing: '0.07em' }}>FINBERT · SENTIMENT BREAKDOWN</div>
                  <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                    {[
                      { label: 'Positive', value: sentiment.positive ?? 0, color: '#22c55e' },
                      { label: 'Neutral', value: sentiment.neutral ?? 0, color: '#f59e0b' },
                      { label: 'Negative', value: sentiment.negative ?? 0, color: '#ef4444' },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: '10px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)', background: 'rgba(255,255,255,0.01)' }}>
                        <div style={{ fontSize: '0.72rem', color: '#cbd5e1' }}>{item.label}</div>
                        <div style={{ marginTop: '6px', fontWeight: 900, color: item.color }}>{pct(item.value)}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ padding: '14px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.25)' }}>
                  <div style={{ fontSize: '0.75rem', color: '#c4b5fd', fontWeight: 800, letterSpacing: '0.07em' }}>GLINER · ENTITY EXTRACTION</div>
                  <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {result.entities.length === 0 ? <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>No entities found.</span> : result.entities.slice(0, 8).map((e, i) => (
                      <span key={`${e.type}-${i}`} style={{ fontSize: '0.75rem', padding: '7px 10px', borderRadius: '999px', border: '1px solid rgba(196,181,253,0.3)', color: '#ddd6fe', background: 'rgba(168,85,247,0.1)' }}>
                        {e.type}: {e.value}{e.confidence !== undefined ? ` (${pct(e.confidence)})` : ''}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Financial NLP Output */}
            <div className="glass-panel" style={{ padding: '28px', background: 'rgba(59,130,246,0.04)', border: '1px solid rgba(59,130,246,0.18)' }}>
              <h4 style={{ marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1rem', fontWeight: 900, color: '#60a5fa' }}>
                <Tag size={18} /> Financial NLP Output
              </h4>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: '#93c5fd', fontWeight: 900, marginBottom: '8px', letterSpacing: '0.05em' }}>DETECTED TERMS</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {financialTerms.length === 0 ? (
                      <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>No financial terms detected yet.</span>
                    ) : financialTerms.slice(0, 14).map((term, index) => (
                      <span key={`${term}-${index}`} style={{ padding: '6px 10px', borderRadius: '999px', background: 'rgba(59,130,246,0.12)', border: '1px solid rgba(59,130,246,0.25)', color: '#bfdbfe', fontSize: '0.75rem', fontWeight: 700 }}>
                        {term}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '0.72rem', color: '#a7f3d0', fontWeight: 900, marginBottom: '8px', letterSpacing: '0.05em' }}>FINANCIAL PARAMETERS</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    {[
                      { label: 'amounts', color: '#38bdf8' },
                      { label: 'rates', color: '#f59e0b' },
                      { label: 'tenures', color: '#34d399' },
                      { label: 'institutions', color: '#c084fc' },
                      { label: 'products', color: '#fb7185' },
                      { label: 'dates', color: '#f97316' },
                    ].map(({ label, color }) => (
                      <div key={label} style={{ padding: '12px', borderRadius: '14px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.07)' }}>
                        <div style={{ fontSize: '0.68rem', color, fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>{label}</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {(financialParameters[label] || []).length === 0 ? (
                            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>None</span>
                          ) : (
                            financialParameters[label].slice(0, 4).map((value, idx) => (
                              <span key={`${label}-${idx}-${value}`} style={{ fontSize: '0.75rem', padding: '5px 8px', borderRadius: '999px', background: 'rgba(255,255,255,0.04)', color: '#e2e8f0' }}>
                                {value}
                              </span>
                            ))
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  <span style={{ padding: '5px 10px', borderRadius: '999px', fontSize: '0.7rem', fontWeight: 800, background: 'rgba(16,185,129,0.2)', color: '#86efac' }}>
                    NLP Topic: {result?.financial_nlp_topic || result?.financial_topic || 'general'}
                  </span>
                  <span style={{ padding: '5px 10px', borderRadius: '999px', fontSize: '0.7rem', fontWeight: 800, background: 'rgba(168,85,247,0.2)', color: '#ddd6fe' }}>
                    Topic Confidence: {Math.round((result?.financial_nlp_topic_confidence || 0) * 100)}%
                  </span>
                </div>
              </div>
            </div>

            {/* RAG Memory Output */}
            <div className="glass-panel" style={{ padding: '28px', background: 'rgba(59,130,246,0.04)', border: '1px solid rgba(59,130,246,0.18)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
                <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1rem', fontWeight: 900, color: '#60a5fa' }}>
                  <Sparkles size={18} /> RAG Memory Output
                </h4>
                <button
                  onClick={() => result && fetchRagInsight(result)}
                  style={{ padding: '8px 12px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.12)', background: 'rgba(255,255,255,0.03)', color: '#e2e8f0', cursor: 'pointer', fontWeight: 700, fontSize: '0.75rem' }}
                >
                  REFRESH
                </button>
              </div>

              {ragLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#93c5fd' }}>
                  <Loader2 className="spin" size={16} />
                  <span style={{ fontSize: '0.9rem' }}>Fetching related sessions...</span>
                </div>
              )}

              {!ragLoading && ragError && (
                <p style={{ margin: 0, color: '#fda4af', fontSize: '0.9rem', lineHeight: 1.5 }}>{ragError}</p>
              )}

              {!ragLoading && !ragError && ragInsight && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  <p style={{ margin: 0, color: '#f8fafc', lineHeight: 1.65, fontSize: '0.95rem' }}>
                    {ragInsight.contextual_insight || 'No RAG insight generated yet.'}
                  </p>

                  <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                    <span style={{ padding: '5px 10px', borderRadius: '999px', fontSize: '0.7rem', fontWeight: 800, background: 'rgba(96,165,250,0.2)', color: '#bfdbfe' }}>
                      Retrieved: {ragRows.length}
                    </span>
                    <span style={{ padding: '5px 10px', borderRadius: '999px', fontSize: '0.7rem', fontWeight: 800, background: 'rgba(16,185,129,0.2)', color: '#86efac' }}>
                      Trajectory: {ragInsight.risk_trajectory || 'unknown'}
                    </span>
                    <span style={{ padding: '5px 10px', borderRadius: '999px', fontSize: '0.7rem', fontWeight: 800, background: 'rgba(168,85,247,0.2)', color: '#ddd6fe' }}>
                      Confidence: {ragInsight.confidence || 'unknown'}
                    </span>
                  </div>

                  {!!(ragInsight.pattern_summary || []).length && (
                    <div>
                      <div style={{ fontSize: '0.72rem', color: '#93c5fd', fontWeight: 900, marginBottom: '8px', letterSpacing: '0.05em' }}>PATTERN SUMMARY</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {(ragInsight.pattern_summary || []).map((point, idx) => (
                          <div key={idx} style={{ fontSize: '0.86rem', color: '#cbd5e1', lineHeight: 1.45 }}>
                            • {point}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {!!(ragInsight.recommended_next_focus || []).length && (
                    <div>
                      <div style={{ fontSize: '0.72rem', color: '#a7f3d0', fontWeight: 900, marginBottom: '8px', letterSpacing: '0.05em' }}>NEXT FOCUS</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {(ragInsight.recommended_next_focus || []).map((point, idx) => (
                          <div key={idx} style={{ fontSize: '0.86rem', color: '#bbf7d0', lineHeight: 1.45 }}>
                            • {point}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {!!ragRows.length && (
                    <div style={{ marginTop: '4px' }}>
                      <div style={{ fontSize: '0.72rem', color: '#fcd34d', fontWeight: 900, marginBottom: '10px', letterSpacing: '0.05em' }}>RETRIEVED SESSIONS</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {ragRows.slice(0, 5).map((row, idx) => (
                          <div key={idx} style={{ padding: '12px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
                            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
                              <span style={{ fontSize: '0.68rem', color: '#93c5fd' }}>#{idx + 1}</span>
                              <span style={{ fontSize: '0.68rem', color: '#f1f5f9' }}>{row.conversation_id || 'unknown'}</span>
                              <span style={{ fontSize: '0.68rem', color: '#a7f3d0' }}>score {(row.score ?? 0).toFixed(2)}</span>
                              <span style={{ fontSize: '0.68rem', color: '#fcd34d' }}>{row.metadata?.risk_level || 'UNKNOWN'}</span>
                              <span style={{ fontSize: '0.68rem', color: '#c4b5fd' }}>{row.metadata?.topic || 'general'}</span>
                            </div>
                            <div style={{ fontSize: '0.84rem', color: '#cbd5e1', lineHeight: 1.45 }}>
                              {extractSummary(row.document)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* World-Class Risk Analysis */}
            <div className="glass-panel" style={{ padding: '40px', background: `linear-gradient(to bottom, ${rc(result.risk_level)}10, transparent)` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '24px' }}>
                  <ShieldCheck size={40} style={{ color: rc(result.risk_level) }} />
                  <div>
                      <h4 style={{ fontSize: '0.8rem', opacity: 0.6, textTransform: 'uppercase', fontWeight: 900 }}>Risk Profile</h4>
                      <div style={{ fontSize: '2.6rem', fontWeight: 900, color: rc(result.risk_level), letterSpacing: '-0.06em' }}>{result.risk_level}</div>
                  </div>
              </div>
              <p style={{ fontSize: '0.95rem', opacity: 0.8, lineHeight: 1.6, borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '20px' }}>{result.risk_assessment}</p>
            </div>

            {/* High-Precision GLiNER Entities */}
            <div className="glass-panel" style={{ padding: '32px', flex: 1 }}>
              <h4 style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px', fontSize: '1rem', fontWeight: 900, color: '#3b82f6' }}><Tag size={20} /> Precision Entity Wall</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {result.entities.length === 0 ? <p style={{ opacity: 0.3, fontSize: '0.9rem' }}>No technical objects captured.</p> :
                 result.entities.map((e,i)=>(
                  <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.015)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.04)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderLeft: `4px solid ${e.type==='AMOUNT' ? '#3b82f6' : '#a855f7'}` }}>
                    <div>
                      <div style={{ fontSize: '0.7rem', opacity: 0.4, fontWeight: 900, textTransform: 'uppercase', marginBottom: '4px' }}>{e.type}</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700 }}>{e.value}</div>
                    </div>
                    <TrendingUp size={16} opacity={0.3} />
                  </div>
                ))}
              </div>
            </div>

            <button onClick={()=>setMode('idle')} style={{ padding: '22px', borderRadius: '24px', background: '#3b82f6', color: 'white', border: 'none', fontWeight: 900, cursor: 'pointer', fontSize: '1.1rem', letterSpacing: '0.05em', boxShadow: '0 15px 30px -10px #3b82f6' }}>NEW STRATEGIC ANALYSIS</button>
          </div>

        </div>
      )}

      {/* Global CSS for McKinsey Black Theme */}
      <style>{`
        body { background: black !important; }
        .glass-panel { background: rgba(255, 255, 255, 0.01); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 32px; }
        .spin { animation: rotate 2s linear infinite; }
        @keyframes rotate { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
