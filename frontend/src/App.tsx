import { useEffect, useMemo, useRef, useState } from 'react';
import { Copy, Mic, Pencil, Shield, Download } from 'lucide-react';
import LandingPage from './pages/LandingPage.tsx';
import TravelConnectSignin, { type TravelConnectAuthMode } from './components/ui/travel-connect-signin';
import { AppSidebar } from './components/app-sidebar';
import { SidebarProvider, SidebarTrigger } from './components/ui/sidebar';
import { PromptBox } from './components/ui/chatgpt-prompt-input';

const exportPDF = (d: AnalysisResult) => {
  const printWindow = window.open('', '_blank');
  if (!printWindow) return;
  printWindow.document.write(`
    <html>
      <head>
        <title>FinFlux Advisor Report</title>
        <style>
          @page { size: A4 portrait; margin: 20mm; }
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", default; color: #1e293b; line-height: 1.6; font-size: 11pt; padding: 0; margin: 0; background: #fff; }
          h1 { color: #020617; font-size: 28pt; letter-spacing: -1px; margin-bottom: 0; border-bottom: 3px solid #0f172a; padding-bottom: 10px; }
          h2 { color: #0f172a; font-size: 16pt; margin-top: 35px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; }
          .page-break { page-break-before: always; }
          .cover { display: flex; flex-direction: column; min-height: 90vh; justify-content: center; }
          .brand { font-size: 40pt; font-weight: 800; background: linear-gradient(180deg, #1e293b 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -2px; }
          .meta { margin-top: 40px; border-top: 2px solid #e2e8f0; padding-top: 20px; font-size: 12pt; }
          .meta-row { display: flex; justify-content: space-between; margin-bottom: 10px; }
          .executive { padding: 30px; background: #f8fafc; border-left: 4px solid #059669; font-size: 13pt; margin-top: 40px; }
          .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 20px; }
          .card { border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; }
          .tag { display: inline-block; padding: 4px 10px; background: #f1f5f9; border-radius: 4px; font-weight: 600; font-size: 10pt; }
          .tag.high { background: #fee2e2; color: #991b1b; }
          .tag.low { background: #d1fae5; color: #065f46; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; }
          th, td { text-align: left; padding: 12px; border-bottom: 1px solid #e2e8f0; }
          th { font-weight: 600; color: #64748b; font-size: 10pt; text-transform: uppercase; }
          .expert-wall { font-family: "Georgia", serif; font-size: 11pt; line-height: 1.8; color: #334155; white-space: pre-wrap; margin-top: 20px; }
        </style>
      </head>
      <body>
        <!-- Page 1: Cover & Executive Summary -->
        <div class="cover">
          <div class="brand">finflux</div>
          <h1 style="border: none;">Client Insight Report</h1>
          
          <div class="meta">
            <div class="meta-row"><strong>Report ID:</strong> <span>${d.conversation_id}</span></div>
            <div class="meta-row"><strong>Extracted:</strong> <span>${new Date(d.timestamp || Date.now()).toLocaleString()}</span></div>
            <div class="meta-row"><strong>Audio Language:</strong> <span>${d.language || 'EN'} (Conf: ${Math.round((d.language_confidence || 0) * 100)}%)</span></div>
          </div>
          
          <div class="executive">
            <p style="text-transform: uppercase; font-size: 10pt; font-weight: bold; color: #059669; margin: 0 0 10px 0;">Executive Summary</p>
            ${d.executive_summary || 'No summary available.'}
          </div>
        </div>

        <!-- Page 2: Risk Audit & Metrics -->
        <div class="page-break">
          <h1>Risk Audit & NLP Extraction</h1>
          <div class="grid">
            <div class="card">
              <h2>Key Identifiers</h2>
              <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Risk Level</td><td><span class="tag ${String(d.risk_level).toLowerCase()}">${d.risk_level || 'N/A'}</span></td></tr>
                <tr><td>Dominant Sentiment</td><td><span class="tag">${d.financial_sentiment || 'Neutral'}</span></td></tr>
                <tr><td>Strategic Intent</td><td>${d.strategic_intent || 'N/A'}</td></tr>
              </table>
            </div>
            <div class="card">
              <h2>Data Pipeline Metrics</h2>
              <table>
                <tr><th>Agent</th><th>Score</th></tr>
                <tr><td>Speech-to-Text</td><td>${Math.round((d.quality_metrics?.asr_confidence || 0) * 100)}%</td></tr>
                <tr><td>NER Coverage</td><td>${Math.round(d.quality_metrics?.ner_coverage_pct || 0)}%</td></tr>
                <tr><td>ROUGE-1 Recall</td><td>${Math.round((d.quality_metrics?.rouge1_recall || 0) * 100)}%</td></tr>
              </table>
            </div>
          </div>
          <h2>Identified Financial Entities</h2>
          <table>
            <tr><th>Entity Type</th><th>Extracted Value</th></tr>
            ${(d.entities || []).map(e => `<tr><td><strong>${e.type}</strong></td><td>${e.value}</td></tr>`).join('') || '<tr><td colspan="2">No entities found.</td></tr>'}
          </table>
          
          <h2>Raw Transcript</h2>
          <p style="padding: 15px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;">${d.transcript || d.raw_asr_text || 'No transcript available.'}</p>
        </div>

        <!-- Page 3: Expert Reasoning Walls -->
        <div class="page-break">
          <h1>Expert Analyst Reasoning</h1>
          <div class="expert-wall">${d.expert_reasoning_points || 'No expert reasoning provided.'}</div>
          
          <h2 style="margin-top: 40px;">Future Outlook & Gearing</h2>
          <div class="expert-wall" style="padding: 20px; background: #f1f5f9; border-left: 4px solid #3b82f6;">${d.future_gearing || 'Monitor situation actively.'}</div>
        </div>
        
        <script>
          window.onload = () => { window.print(); window.setTimeout(() => window.close(), 500); }
        </script>
      </body>
    </html>
  `);
  printWindow.document.close();
};

type ViewMode = 'chat' | 'insights' | 'settings';

interface TopicScore { topic: string; score: number }
interface SentimentBreakdown { positive?: number; neutral?: number; negative?: number }
interface Entity { type: string; value: string; confidence?: number }
interface QualityMetrics {
  asr_confidence?: number;
  ner_coverage_pct?: number;
  rouge1_recall?: number;
  entity_alignment_pct?: number;
  language_confidence?: number;
  financial_relevance_score?: number;
  overall_quality_score?: number;
  quality_tier?: string;
}
interface AnalysisResult {
  conversation_id: string;
  chat_thread_id?: string;
  response_mode?: 'analysis' | 'financial_inquiry' | 'general_conversation';
  timestamp: string;
  language?: string;
  language_confidence?: number;
  financial_topic: string;
  risk_level: string;
  financial_sentiment: string;
  confidence_score: number;
  executive_summary: string;
  transcript: string;
  assistant_text?: string;
  strategic_intent?: string;
  future_gearing?: string;
  risk_assessment?: string;
  expert_reasoning_points?: string;
  timing?: { total_s?: number };
  topic_top3?: TopicScore[];
  sentiment_breakdown?: SentimentBreakdown;
  quality_metrics?: QualityMetrics;
  entities: Entity[];
  key_insights?: string[];
  raw_asr_text?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  conversationId?: string;
  attachedResult?: AnalysisResult;
}

interface ThreadSummary {
  thread_id: string;
  last_timestamp: string;
  preview: string;
  topic: string;
  risk_level: string;
  count: number;
}

const riskColor = (level: string) => {
  switch ((level || '').toUpperCase()) {
    case 'CRITICAL': return '#ef4444';
    case 'HIGH': return '#f97316';
    case 'MEDIUM': return '#eab308';
    case 'LOW': return '#22c55e';
    default: return '#64748b';
  }
};

const API_BASE = 'http://localhost:8000';

/* Lightweight markdown →  JSX for LLM output: **bold**, numbered lists, line breaks */
function renderMarkdown(text: string): React.ReactNode[] {
  if (!text) return [];
  const lines = text.split('\n');
  const nodes: React.ReactNode[] = [];
  let listItems: React.ReactNode[] = [];
  const flushList = () => {
    if (listItems.length > 0) {
      nodes.push(<ol key={`ol-${nodes.length}`} className="md-list">{listItems}</ol>);
      listItems = [];
    }
  };
  lines.forEach((line, idx) => {
    const trimmed = line.trim();
    if (!trimmed) { flushList(); return; }
    const listMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (listMatch) {
      listItems.push(<li key={idx}>{parseBold(listMatch[2])}</li>);
      return;
    }
    const bulletMatch = trimmed.match(/^[-•]\s+(.+)$/);
    if (bulletMatch) {
      listItems.push(<li key={idx}>{parseBold(bulletMatch[1])}</li>);
      return;
    }
    flushList();
    nodes.push(<p key={idx} className="md-para">{parseBold(trimmed)}</p>);
  });
  flushList();
  return nodes;
}
function parseBold(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function encodeWavBlob(samples: Float32Array, sr: number): Blob {
  const bps = 16;
  const nc = 1;
  const ba = nc * (bps / 8);
  const dl = samples.length * ba;
  const buf = new ArrayBuffer(44 + dl);
  const view = new DataView(buf);
  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dl, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, nc, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * ba, true);
  view.setUint16(32, ba, true);
  view.setUint16(34, bps, true);
  writeString(36, 'data');
  view.setUint32(40, dl, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [token, setToken] = useState<string>('');
  const [username, setUsername] = useState<string>('');
  const [authMode, setAuthMode] = useState<TravelConnectAuthMode>('signin');
  const [authError, setAuthError] = useState('');
  const [authNotice, setAuthNotice] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const [view, setView] = useState<ViewMode>('chat');
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isRecordPaused, setIsRecordPaused] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingLevels, setRecordingLevels] = useState<number[]>(Array.from({ length: 28 }, () => 0.06));
  const [error, setError] = useState('');
  const [copiedMessageId, setCopiedMessageId] = useState('');
  const [editingConversationId, setEditingConversationId] = useState('');
  const [voiceReplyOn] = useState(false);
  const [selectedInsightConversationId, setSelectedInsightConversationId] = useState('');
  const [selectedDetail, setSelectedDetail] = useState<AnalysisResult | null>(null);
  const [themeMode, setThemeMode] = useState<'auto' | 'dark' | 'light'>('dark');
  const [fontSizePx, setFontSizePx] = useState<number>(14);
  const [textColor, setTextColor] = useState<string>('#e2e8f0');
  const [audioLanguage, setAudioLanguage] = useState<'auto' | 'hi' | 'en'>('auto');
  const [ttsVoiceProfile, setTtsVoiceProfile] = useState<'auto' | 'female' | 'male'>('auto');

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const levelFrameRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const pcmBufferRef = useRef<Float32Array[]>([]);

  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--app-font-size', `${fontSizePx}px`);
    root.style.setProperty('--app-text-color', textColor);
    root.style.setProperty('--text-primary', textColor);
  }, [fontSizePx, textColor]);

  useEffect(() => {
    const root = document.documentElement;
    const media = window.matchMedia('(prefers-color-scheme: dark)');

    const applyTheme = () => {
      if (themeMode === 'auto') {
        root.setAttribute('data-theme', media.matches ? 'dark' : 'light');
      } else {
        root.setAttribute('data-theme', themeMode);
      }
    };

    applyTheme();

    if (themeMode !== 'auto') return;

    const onSystemThemeChange = () => applyTheme();
    if (media.addEventListener) {
      media.addEventListener('change', onSystemThemeChange);
      return () => media.removeEventListener('change', onSystemThemeChange);
    }

    media.addListener(onSystemThemeChange);
    return () => media.removeListener(onSystemThemeChange);
  }, [themeMode]);

  useEffect(() => {
    if (error !== 'Recording canceled.') return;
    const timeoutId = window.setTimeout(() => {
      setError((prev) => (prev === 'Recording canceled.' ? '' : prev));
    }, 2000);
    return () => window.clearTimeout(timeoutId);
  }, [error]);

  useEffect(() => {
    if (!token) return;
    void loadHistory();
    void loadThreads();
  }, [token]);

  const authFetch = async (path: string, options: RequestInit = {}) => {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${token}`,
      },
    });
    if (res.status === 401) {
      handleSignOut();
      throw new Error('Session expired. Please sign in again.');
    }
    return res;
  };

  const loadHistory = async () => {
    if (!token) return;
    try {
      const res = await authFetch('/api/results');
      const data = await res.json();
      const rows: AnalysisResult[] = data.results || [];
      setHistory(rows);
    } catch (e) {
      setError(String(e));
    }
  };

  const loadThreads = async () => {
    if (!token) return;
    try {
      const res = await authFetch('/api/threads');
      const data = await res.json();
      const rows: ThreadSummary[] = data.results || [];
      setThreads(rows);
      // Don't auto-open any thread; user should see empty chat after login
    } catch (e) {
      setError(String(e));
    }
  };

  const streamAssistantMessage = (messageId: string, fullText: string, attachedResult?: AnalysisResult) => new Promise<void>((resolve) => {
    let index = 0;
    const tick = () => {
      index += 2;
      const nextText = fullText.slice(0, index);
      setMessages((prev) => prev.map((m) => (m.id === messageId ? { ...m, text: nextText, attachedResult } : m)));
      if (index < fullText.length) {
        window.setTimeout(tick, 12);
      } else {
        resolve();
      }
    };
    tick();
  });

  const detectSpeakLanguage = (result?: AnalysisResult): 'hi-IN' | 'en-IN' => {
    const lang = String(result?.language || '').toLowerCase();
    if (lang.startsWith('hi') || lang.includes('hindi')) return 'hi-IN';
    return 'en-IN';
  };

  const normalizeSpeechText = (text: string): string => {
    const src = String(text || '').trim();
    if (!src) return '';
    return src
      .replace(/₹\s?([0-9][0-9,]*)/g, '$1 rupees')
      .replace(/INR\s?([0-9][0-9,]*)/gi, '$1 rupees')
      .replace(/\b([0-9]{1,3}(?:,[0-9]{3})+)\b/g, (_, g1) => String(g1).replace(/,/g, ' '))
      .replace(/\s+/g, ' ')
      .trim();
  };

  const firstSentence = (text: string): string => {
    const t = String(text || '').trim();
    if (!t) return '';
    const m = t.match(/^[^.!?]+[.!?]?/);
    return m ? m[0].trim() : t;
  };

  const buildSpeakSummary = (text: string, result?: AnalysisResult): string => {
    if (!result || result.response_mode !== 'analysis') {
      return normalizeSpeechText(text);
    }
    const summary = firstSentence(result.executive_summary || text || '');
    const risk = (result.risk_level || 'LOW').toUpperCase();
    const topic = result.financial_topic || 'financial discussion';
    const next = firstSentence(result.future_gearing || 'Monitor cash flow and commitment follow-through.');
    return normalizeSpeechText(`Quick insight: Topic is ${topic}. Risk is ${risk}. ${summary} Next: ${next}`);
  };

  const pickVoice = (langTag: 'hi-IN' | 'en-IN') => {
    if (!('speechSynthesis' in window)) return null;
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;

    const primary = voices.filter((v) => v.lang?.toLowerCase().startsWith(langTag.slice(0, 2).toLowerCase()));
    const pool = primary.length > 0 ? primary : voices;

    if (ttsVoiceProfile === 'female') {
      const female = pool.find((v) => /female|zira|susan|aria|heera|veena|siri/i.test(v.name));
      if (female) return female;
    }
    if (ttsVoiceProfile === 'male') {
      const male = pool.find((v) => /male|david|mark|alex|ravi|rahul|george/i.test(v.name));
      if (male) return male;
    }

    return pool[0] || voices[0] || null;
  };

  const speak = (text: string, result?: AnalysisResult) => {
    if (!text.trim() || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(buildSpeakSummary(text, result));
    const langTag = detectSpeakLanguage(result);
    utterance.lang = langTag;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    const voice = pickVoice(langTag);
    if (voice) utterance.voice = voice;
    window.speechSynthesis.speak(utterance);
  };

  const openHistoryThread = async (threadId: string) => {
    try {
      const res = await authFetch(`/api/threads/${threadId}/messages`);
      const data = await res.json();
      const rows = (data.results || []) as Array<{ id: string; role: 'user' | 'assistant'; text: string; conversation_id?: string; attached_result?: AnalysisResult }>;
      setMessages(rows.map((r) => ({
        id: r.id,
        role: r.role,
        text: r.text,
        conversationId: r.conversation_id,
        attachedResult: r.attached_result,
      })));
      const latestAnalysis = [...rows].reverse().find((r) => r.role === 'assistant' && r.attached_result)?.attached_result;
      setSelectedDetail(latestAnalysis || null);
      setActiveThreadId(threadId);
      setView('chat');
    } catch (e) {
      setError(String(e));
    }
  };

  const focusComposer = () => {
    window.setTimeout(() => {
      const textarea = document.querySelector('.prompt-textarea') as HTMLTextAreaElement | null;
      textarea?.focus();
      textarea?.setSelectionRange(textarea.value.length, textarea.value.length);
    }, 0);
  };

  const getPromptByConversationId = (conversationId: string) => {
    const match = [...messages].reverse().find((m) => m.role === 'user' && m.conversationId === conversationId);
    return match?.text || '';
  };

  const startConversationEdit = (conversationId: string, fallbackPrompt = '') => {
    let prompt = getPromptByConversationId(conversationId) || fallbackPrompt;
    if (prompt === 'Voice message sent' && fallbackPrompt) {
      prompt = fallbackPrompt;
    }
    if (!prompt.trim()) {
      setError('Original prompt not found for this summary.');
      return;
    }
    setEditingConversationId(conversationId);
    setInput(prompt);
    setView('chat');
    setError('');
    focusComposer();
  };

  const copyPrompt = async (messageId: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text || '');
      setCopiedMessageId(messageId);
      window.setTimeout(() => {
        setCopiedMessageId((prev) => (prev === messageId ? '' : prev));
      }, 1400);
    } catch {
      setError('Unable to copy prompt.');
    }
  };

  const editPrompt = (text: string, conversationId?: string) => {
    if (conversationId) {
      startConversationEdit(conversationId, text);
      return;
    }
    setInput(text || '');
    setView('chat');
    focusComposer();
  };

  const cancelConversationEdit = () => {
    setEditingConversationId('');
    setInput('');
  };

  const saveConversationEdit = async () => {
    const transcript = input.trim();
    if (!editingConversationId || !transcript || isSending || !token) return;

    setIsSending(true);
    setError('');
    try {
      const res = await authFetch(`/api/conversations/${editingConversationId}/transcript`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript, reanalyze: true }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Update failed' }));
        throw new Error(body.detail || 'Update failed');
      }

      const payload = await res.json();
      const updated = payload.conversation as AnalysisResult | undefined;
      if (!updated) throw new Error('Invalid update response');

      const updatedAssistantText = updated.executive_summary || 'Analysis complete.';
      const updatedResult: AnalysisResult = { ...updated, response_mode: 'analysis' };

      setMessages((prev) => prev.map((m) => {
        if (m.conversationId !== editingConversationId) return m;
        if (m.role === 'user') return { ...m, text: transcript };
        if (m.role === 'assistant') {
          return {
            ...m,
            text: updatedAssistantText,
            attachedResult: updatedResult,
          };
        }
        return m;
      }));

      setSelectedDetail((prev) => {
        if (!prev || prev.conversation_id !== editingConversationId) return prev;
        return updatedResult;
      });

      setInput('');
      setEditingConversationId('');
      await loadHistory();
      await loadThreads();
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSending(false);
    }
  };

  const handleAuth = async (payload: { mode: TravelConnectAuthMode; email: string; username: string; password: string }) => {
    setAuthLoading(true);
    setAuthError('');
    setAuthNotice('');
    try {
      const endpoint = payload.mode === 'signin' ? '/api/auth/login' : '/api/auth/signup';
      const requestBody = payload.mode === 'signin'
        ? { username: payload.email.trim(), password: payload.password }
        : { username: payload.email.trim(), display_name: payload.username.trim(), password: payload.password };
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Auth failed' }));
        throw new Error(err.detail || 'Auth failed');
      }
      const data = await res.json();
      if (!data.access_token) {
        setAuthNotice(data.message || 'Verification email sent. Confirm your email, then sign in.');
        setAuthMode('signin');
        return;
      }
      setToken(data.access_token);
      setUsername(data.username);
      setAuthError('');
      setAuthNotice('');
      setView('chat');
    } catch (e) {
      setAuthError(String(e));
      setAuthNotice('');
    } finally {
      setAuthLoading(false);
    }
  };

  const deleteThread = async (threadId: string) => {
    if (!token) return;
    try {
      const res = await authFetch(`/api/threads/${threadId}`, { method: 'DELETE' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Delete failed' }));
        throw new Error(body.detail || 'Failed to delete thread');
      }
      // Remove thread from list and clear view if it was active
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
      if (activeThreadId === threadId) {
        setActiveThreadId('');
        setMessages([]);
        setSelectedDetail(null);
        setView('chat');
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const handleSignOut = () => {
    setToken('');
    setUsername('');
    setHistory([]);
    setThreads([]);
    setMessages([]);
    setActiveThreadId('');
    setEditingConversationId('');
    setAuthError('');
    setAuthNotice('');
  };

  const insights = useMemo(() => {
    const total = history.length;
    const avgConfidence = total === 0 ? 0 : history.reduce((s, r) => s + (r.confidence_score || 0), 0) / total;
    const avgLatency = total === 0 ? 0 : history.reduce((s, r) => s + (r.timing?.total_s || 0), 0) / total;

    const riskMap: Record<string, number> = { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 };
    const sentimentMap: Record<string, number> = { positive: 0, neutral: 0, negative: 0 };
    const topicMap: Record<string, number> = {};

    for (const row of history) {
      const risk = (row.risk_level || 'LOW').toUpperCase();
      if (riskMap[risk] !== undefined) riskMap[risk] += 1;
      const sent = (row.financial_sentiment || 'neutral').toLowerCase();
      if (sent.includes('pos')) sentimentMap.positive += 1;
      else if (sent.includes('neg')) sentimentMap.negative += 1;
      else sentimentMap.neutral += 1;
      const topic = (row.financial_topic || 'general').toLowerCase();
      topicMap[topic] = (topicMap[topic] || 0) + 1;
    }

    const topTopics = Object.entries(topicMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    return { total, avgConfidence, avgLatency, riskMap, sentimentMap, topTopics };
  }, [history]);

  const selectedInsightConversation = useMemo(
    () => history.find((h) => h.conversation_id === selectedInsightConversationId) || null,
    [history, selectedInsightConversationId],
  );


  if (showLanding) {
    return <LandingPage onGetStarted={() => {
      setAuthMode('signin');
      setShowLanding(false);
      window.scrollTo(0, 0);
    }} />;
  }

  const createNewChat = () => {
    const newThreadId = `thr_${(crypto?.randomUUID?.() || Date.now().toString()).replace(/-/g, '').slice(0, 12)}`;
    setActiveThreadId(newThreadId);
    setMessages([]);
    setInput('');
    setView('chat');
    setSelectedDetail(null);
  };

  const onComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handlePromptSubmit();
    }
  };

  const sendText = async () => {
    const text = input.trim();
    if (!text || !token || isSending) return;

    const userMessageId = `u-${Date.now()}`;
    const userMessage: ChatMessage = { id: userMessageId, role: 'user', text };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsSending(true);
    setError('');

    try {
      const res = await authFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, thread_id: activeThreadId || undefined }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(body.detail || 'Request failed');
      }
      const data: AnalysisResult = await res.json();
      const assistantId = `a-${Date.now()}`;
      const assistantText = data.executive_summary || data.assistant_text || 'Analysis complete.';
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        text: '',
        conversationId: data.conversation_id,
      };
      setMessages((prev) => prev.map((m) => (m.id === userMessageId ? { ...m, conversationId: data.conversation_id } : m)));
      setMessages((prev) => [...prev, assistantMessage]);
      await streamAssistantMessage(assistantId, assistantText, data);
      setSelectedDetail(data);
      if (voiceReplyOn) speak(assistantText, data);
      setActiveThreadId(data.chat_thread_id || activeThreadId);
      await loadHistory();
      await loadThreads();
      setUploadedFiles([]);
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSending(false);
    }
  };

  const analyzeAudioFile = async (file: File, userLabel: string) => {
    if (!token || isSending) return;

    const userMessageId = `u-${Date.now()}`;
    const userMessage: ChatMessage = { id: userMessageId, role: 'user', text: userLabel };
    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);
    setError('');

    try {
      const fd = new FormData();
      fd.append('file', file);
      if (activeThreadId) fd.append('thread_id', activeThreadId);
      if (audioLanguage !== 'auto') fd.append('asr_language', audioLanguage);
      const res = await authFetch('/api/analyze', { method: 'POST', body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Audio analyze failed' }));
        throw new Error(body.detail || 'Audio analyze failed');
      }
      const data: AnalysisResult = await res.json();
      const assistantId = `a-${Date.now()}`;
      const assistantText = data.executive_summary || data.assistant_text || 'Audio analysis complete.';
      setMessages((prev) => [...prev, {
        id: assistantId,
        role: 'assistant',
        text: '',
        conversationId: data.conversation_id,
      }]);
      setMessages((prev) => prev.map((m) => (m.id === userMessageId ? { ...m, conversationId: data.conversation_id } : m)));
      await streamAssistantMessage(assistantId, assistantText, data);
      setSelectedDetail(data);
      setActiveThreadId(data.chat_thread_id || activeThreadId);
      if (voiceReplyOn) speak(assistantText, data);
      await loadHistory();
      await loadThreads();
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSending(false);
    }
  };

  const startRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      analyserRef.current = analyser;
      src.connect(analyser);
      const sn = ctx.createScriptProcessor(4096, 1, 1);
      pcmBufferRef.current = [];
      sn.onaudioprocess = (e) => {
        if (!isRecordPaused) {
          pcmBufferRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        }
      };
      src.connect(sn);
      sn.connect(ctx.destination);
      scriptNodeRef.current = sn;

      const animateLevels = () => {
        const a = analyserRef.current;
        if (!a) return;
        const arr = new Uint8Array(a.frequencyBinCount);
        a.getByteFrequencyData(arr);
        const step = Math.max(1, Math.floor(arr.length / 28));
        const next = Array.from({ length: 28 }, (_, i) => {
          const start = i * step;
          const end = Math.min(arr.length, start + step);
          let sum = 0;
          for (let j = start; j < end; j++) sum += arr[j];
          const avg = end > start ? sum / (end - start) : 0;
          return Math.max(0.06, Math.min(1, avg / 255));
        });
        setRecordingLevels(next);
        levelFrameRef.current = window.requestAnimationFrame(animateLevels);
      };

      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = window.setInterval(() => setRecordingSeconds((v) => v + 1), 1000);
      setRecordingSeconds(0);
      animateLevels();
      setIsRecording(true);
      setIsRecordPaused(false);
      setError('');
    } catch (e) {
      setError(`Microphone error: ${String(e)}`);
    }
  };

  const cleanupRecording = () => {
    if (levelFrameRef.current !== null) {
      window.cancelAnimationFrame(levelFrameRef.current);
      levelFrameRef.current = null;
    }
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    scriptNodeRef.current?.disconnect();
    scriptNodeRef.current = null;
    analyserRef.current = null;
    if (audioCtxRef.current) {
      void audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setRecordingLevels(Array.from({ length: 28 }, () => 0.06));
  };

  const togglePauseRecord = () => {
    if (!isRecording) return;
    setIsRecordPaused((v) => !v);
  };

  const cancelRecord = () => {
    if (!isRecording) return;
    cleanupRecording();
    pcmBufferRef.current = [];
    setIsRecording(false);
    setRecordingSeconds(0);
    setError('Recording canceled.');
  };

  const stopRecord = async () => {
    if (!isRecording) return;
    setIsRecording(false);
    cleanupRecording();

    const chunks = pcmBufferRef.current;
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    if (total === 0) {
      setError('No audio captured. Please record again.');
      setRecordingSeconds(0);
      return;
    }
    const merged = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }

    const file = new File([encodeWavBlob(merged, 16000)], 'voice.wav', { type: 'audio/wav' });
    await analyzeAudioFile(file, 'Voice message sent');
    setRecordingSeconds(0);
  };

  const handlePromptSubmit = async () => {
    if (isRecording) return;

    if (editingConversationId && input.trim()) {
      await saveConversationEdit();
      return;
    }

    if (input.trim()) {
      await sendText();
      return;
    }

    if (uploadedFiles.length > 0) {
      const audioFile = uploadedFiles.find((file) => file.type.startsWith('audio/'));
      if (!audioFile) {
        setError('Only audio files are currently supported for direct file analysis.');
        return;
      }
      await analyzeAudioFile(audioFile, `Uploaded audio: ${audioFile.name}`);
      setUploadedFiles([]);
    }
  };

  if (!token) {
    return (
      <TravelConnectSignin
        mode={authMode}
        onModeChange={setAuthMode}
        onSubmit={handleAuth}
        loading={authLoading}
        error={authError}
        notice={authNotice}
      />
    );
  }

  return (
    <SidebarProvider>
      <div className="chat-shell">
        <AppSidebar
          view={view}
          onViewChange={setView}
          onNewChat={createNewChat}
          threads={threads}
          activeThreadId={activeThreadId}
          onOpenThread={openHistoryThread}
          onDeleteThread={deleteThread}
          username={username}
          onSignOut={handleSignOut}
        />

        <main className="chat-main">
        <div className="chat-main-header">
          <SidebarTrigger />
        </div>
        {view === 'chat' && (
          <>
            <div className={`chat-workspace ${messages.length === 0 ? 'no-analysis' : ''}`}>
              <div className="chat-left-pane">
                <div className="messages-wrap">
              {messages.length === 0 && (
                <div className="empty-state">
                  <h2>Record or type a financial conversation to begin.</h2>
                  <p>FinFlux will transcribe, detect financial entities, and generate structured insights.</p>
                  <div className="empty-audio-cta">
                    <button
                      type="button"
                      className="prompt-icon-btn prompt-mic-btn empty-mic-btn"
                      onClick={startRecord}
                      disabled={isSending || isRecording}
                      title="Start recording"
                      aria-label="Start recording"
                    >
                      <Mic size={100} />
                    </button>
                  </div>
                </div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>
                  <div className={`msg-bubble ${msg.attachedResult ? 'has-analysis' : ''}`} onClick={() => msg.attachedResult && setSelectedDetail(msg.attachedResult)}>
                    {msg.role === 'assistant' && msg.attachedResult && (
                        <div className="msg-structured">
                        <div className="msg-structured-header">
                          <span className="msg-topic-badge">{msg.attachedResult.financial_topic || 'Financial Analysis'}</span>
                          <span className="msg-risk-badge" style={{ background: riskColor(msg.attachedResult.risk_level) }}>{msg.attachedResult.risk_level || 'LOW'}</span>
                        </div>
                        <div className="msg-section">
                          <h4>Executive Summary</h4>
                          <div className="md-content">{renderMarkdown(msg.text || msg.attachedResult.executive_summary || 'Analysis complete.')}</div>
                        </div>
                        {msg.attachedResult.key_insights && msg.attachedResult.key_insights.length > 0 && (
                          <div className="msg-section">
                            <h4>Key Insights</h4>
                            <ul>{(msg.attachedResult.key_insights as string[]).map((insight, i) => <li key={i}>{parseBold(insight)}</li>)}</ul>
                          </div>
                        )}
                        {msg.attachedResult.risk_assessment && (
                          <div className="msg-section">
                            <h4>Risk Assessment</h4>
                            <div className="md-content">{renderMarkdown(msg.attachedResult.risk_assessment)}</div>
                          </div>
                        )}
                        {msg.attachedResult.expert_reasoning_points && (
                          <details className="msg-section reasoning-inline">
                            <summary><h4 style={{ display: 'inline' }}>Expert Reasoning</h4></summary>
                            <div className="md-content">{renderMarkdown(msg.attachedResult.expert_reasoning_points)}</div>
                          </details>
                        )}
                        {msg.attachedResult.future_gearing && (
                          <div className="msg-section">
                            <h4>Future Outlook</h4>
                            <div className="md-content">{renderMarkdown(msg.attachedResult.future_gearing)}</div>
                          </div>
                        )}
                        <div className="msg-meta-chips">
                          <span>Intent: {msg.attachedResult.strategic_intent || 'N/A'}</span>
                          <span>Lang: {String(msg.attachedResult.language || 'unknown').toUpperCase()}</span>
                          <span>Sentiment: {msg.attachedResult.financial_sentiment || 'Neutral'}</span>
                          <span>Latency: {(msg.attachedResult.timing?.total_s || 0).toFixed(1)}s</span>
                        </div>
                      </div>
                    )}
                    {msg.role === 'assistant' && !msg.attachedResult && <p>{msg.text}</p>}
                    {msg.role === 'user' && <p>{msg.text}</p>}
                    {msg.role === 'user' && (
                      <div className="msg-tools" onClick={(e) => e.stopPropagation()}>
                        <button className="msg-tool-btn" onClick={() => void copyPrompt(msg.id, msg.text)} title="Copy" aria-label="Copy">
                          <Copy size={13} />
                          <span>{copiedMessageId === msg.id ? 'Copied' : 'Copy'}</span>
                        </button>
                        {msg.text !== 'Voice message sent' && (
                          <button className="msg-tool-btn" onClick={() => editPrompt(msg.text, msg.conversationId)} title="Edit" aria-label="Edit">
                            <Pencil size={13} />
                            <span>Edit</span>
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {error && <div className="error-line">{error}</div>}
                </div>
              </div>

              {messages.length > 0 && (
                <aside className="chat-right-pane electric-border">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 className="gradient-text">Analysis Details</h3>
                    {selectedDetail && (
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="msg-tool-btn" onClick={() => exportPDF(selectedDetail)} title="Export Elite PDF">
                          <Download size={13} />
                          <span>Elite PDF Report</span>
                        </button>
                      </div>
                    )}
                  </div>
                  {!selectedDetail && <p className="detail-placeholder">Select a response to view details.</p>}
                  {selectedDetail && (() => {
                    const d = selectedDetail;
                    const sb = d.sentiment_breakdown || {};
                    const ents = d.entities || [];
                    const qm = d.quality_metrics || {};
                    return (
                      <div className="right-details-card">
                        {/* Transcript (raw audio as-is) */}
                        <div className="detail-section">
                          <h4>Transcript</h4>
                          <div className="transcript-editable">
                            <p className="transcript-text">{d.raw_asr_text || d.transcript || 'No transcript.'}</p>
                            <button className="transcript-edit-btn" onClick={() => startConversationEdit(d.conversation_id, d.raw_asr_text || d.transcript || '')}>
                              <Pencil size={12} /> Edit & Re-analyze
                            </button>
                          </div>
                        </div>

                        {/* Risk & Metrics */}
                        <div className="detail-section">
                          <h4>Risk & Strategy</h4>
                          <div className="detail-metric-grid">
                            <div className="metric-item">
                              <span className="metric-label">Risk Level</span>
                              <span className="metric-value" style={{ color: riskColor(d.risk_level) }}>{d.risk_level || 'LOW'}</span>
                            </div>
                            <div className="metric-item">
                              <span className="metric-label">Confidence</span>
                              <span className="metric-value">{Math.round((d.confidence_score || 0) * 100)}%</span>
                            </div>
                            <div className="metric-item">
                              <span className="metric-label">Sentiment</span>
                              <span className="metric-value">{d.financial_sentiment || 'Neutral'}</span>
                            </div>
                            <div className="metric-item">
                              <span className="metric-label">Intent</span>
                              <span className="metric-value">{d.strategic_intent || 'N/A'}</span>
                            </div>
                          </div>
                          {d.risk_assessment && <p className="detail-sub">{d.risk_assessment}</p>}
                        </div>

                        {/* Sentiment Breakdown */}
                        <div className="detail-section">
                          <h4>Sentiment</h4>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', fontWeight: 'bold', textTransform: 'capitalize', color: d.financial_sentiment?.includes('neg') ? '#ff4ecd' : d.financial_sentiment?.includes('pos') ? '#00ff9c' : '#f8fafc' }}>
                             {d.financial_sentiment || 'Neutral'}
                             <span style={{ color: '#94a3b8', fontWeight: 'normal' }}>({Math.round(Math.max(sb.positive || 0, sb.negative || 0, sb.neutral || 0) * 100)}%)</span>
                          </div>
                        </div>

                        {/* Entities */}
                        {ents.length > 0 && (
                          <div className="detail-section">
                            <h4>Entities</h4>
                            <div className="entity-chips">
                              {ents.slice(0, 10).map((e, i) => (
                                <span key={i} className="entity-chip">
                                  <strong>{e.type}</strong>: {e.value}
                                  {e.confidence && <small> ({Math.round(e.confidence * 100)}%)</small>}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Topic Candidates */}
                        {d.topic_top3 && d.topic_top3.length > 0 && (
                          <div className="detail-section">
                            <h4>Topics</h4>
                            <div className="topic-list">
                              {d.topic_top3.map((t, i) => (
                                <div key={i} className="topic-row">
                                  <span>{t.topic}</span>
                                  <div className="bar-track small"><div className="bar-fill topic" style={{ width: `${Math.round(t.score * 100)}%` }}></div></div>
                                  <span>{Math.round(t.score * 100)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Qwen Expert Reasoning */}
                        {d.expert_reasoning_points && (
                          <details className="detail-section reasoning-collapse">
                            <summary><h4 style={{ display: 'inline' }}>Expert Reasoning</h4></summary>
                            <p className="reasoning-text">{d.expert_reasoning_points}</p>
                          </details>
                        )}

                        {/* Quality Metrics */}
                        {qm.overall_quality_score !== undefined && (
                          <div className="detail-section">
                            <h4>Quality Metrics</h4>
                            <div className="detail-metric-grid">
                              <div className="metric-item">
                                <span className="metric-label">Quality</span>
                                <span className="metric-value">{qm.quality_tier || 'N/A'}</span>
                              </div>
                              <div className="metric-item">
                                <span className="metric-label">ASR Conf.</span>
                                <span className="metric-value">{Math.round((qm.asr_confidence || 0) * 100)}%</span>
                              </div>
                              <div className="metric-item">
                                <span className="metric-label">NER Coverage</span>
                                <span className="metric-value">{Math.round(qm.ner_coverage_pct || (ents.length > 0 ? Math.min(100, ents.length * 20) : 0))}%</span>
                              </div>
                              <div className="metric-item">
                                <span className="metric-label">ROUGE-1</span>
                                <span className="metric-value">{Math.round((qm.rouge1_recall || 0) * 100)}%</span>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Future Outlook */}
                        {d.future_gearing && (
                          <div className="detail-section">
                            <h4>Future Outlook</h4>
                            <p>{d.future_gearing}</p>
                          </div>
                        )}

                        {/* Language */}
                        <div className="detail-section">
                          <h4>Language</h4>
                          <p>
                            {d.language === 'HI' && /[a-z]/i.test(d.transcript || d.raw_asr_text || '') ? 'HINGLISH (HI)' : String(d.language || 'unknown').toUpperCase()} — {Math.round((d.language_confidence || 0) * 100)}% confidence
                          </p>
                          <p className="detail-sub">Latency: {(d.timing?.total_s || 0).toFixed(2)}s</p>
                        </div>
                      </div>
                    );
                  })()}
                </aside>
              )}
            </div>

            <div className="composer">
              {editingConversationId && (
                <div className="composer-edit-banner">
                  <span>Editing this call transcript prompt. Submit to re-run analysis.</span>
                  <button type="button" onClick={cancelConversationEdit}>Cancel edit</button>
                </div>
              )}
              <PromptBox
                value={input}
                onChange={setInput}
                onSubmit={() => void handlePromptSubmit()}
                onKeyDown={onComposerKeyDown}
                isSending={isSending}
                isRecording={isRecording}
                isRecordPaused={isRecordPaused}
                recordingLevels={recordingLevels}
                recordingSeconds={recordingSeconds}
                audioLanguage={audioLanguage}
                onAudioLanguageChange={setAudioLanguage}
                onStartRecord={startRecord}
                onTogglePauseRecord={togglePauseRecord}
                onStopRecord={() => void stopRecord()}
                onCancelRecord={cancelRecord}
                files={uploadedFiles}
                onPickFiles={(files) => {
                  setUploadedFiles((prev) => [...prev, ...files]);
                  setError('');
                }}
                onRemoveFile={(index) => {
                  setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
                }}
              />
            </div>
          </>
        )}

        {view === 'insights' && (
          <div className="insights-wrap">
            <h2>Insights + Investments</h2>
            <div className="insights-grid">
              <div className="insight-card"><h3>Total conversations</h3><p>{insights.total}</p></div>
              <div className="insight-card"><h3>Average confidence</h3><p>{Math.round(insights.avgConfidence * 100)}%</p></div>
              <div className="insight-card"><h3>Average pipeline latency</h3><p>{insights.avgLatency.toFixed(2)}s</p></div>
              <div className="insight-card">
                <h3>Risk distribution</h3>
                <ul>
                  <li>LOW: {insights.riskMap.LOW}</li>
                  <li>MEDIUM: {insights.riskMap.MEDIUM}</li>
                  <li>HIGH: {insights.riskMap.HIGH}</li>
                  <li>CRITICAL: {insights.riskMap.CRITICAL}</li>
                </ul>
              </div>
              <div className="insight-card">
                <h3>Sentiment distribution</h3>
                <ul>
                  <li>Positive: {insights.sentimentMap.positive}</li>
                  <li>Neutral: {insights.sentimentMap.neutral}</li>
                  <li>Negative: {insights.sentimentMap.negative}</li>
                </ul>
              </div>
              <div className="insight-card">
                <h3>Top 5 topics</h3>
                <ul>
                  {insights.topTopics.map(([topic, count]) => <li key={topic}>{topic}: {count}</li>)}
                </ul>
              </div>
              <div className="insight-card">
                <h3>Investment Intent Signals</h3>
                <ul>
                  {insights.topTopics.filter(([topic]) => topic.includes('invest') || topic.includes('sip') || topic.includes('fund')).slice(0, 5).map(([topic, count]) => (
                    <li key={`inv-${topic}`}>{topic}: {count}</li>
                  ))}
                  {insights.topTopics.filter(([topic]) => topic.includes('invest') || topic.includes('sip') || topic.includes('fund')).length === 0 && (
                    <li>No strong investment pattern detected yet.</li>
                  )}
                </ul>
              </div>
            </div>

            <div className="insights-drilldown">
              <h3>Conversation Drill-down</h3>
              <div className="drilldown-table">
                {history.slice(0, 12).map((row) => (
                  <button
                    key={row.conversation_id}
                    className={`drilldown-row ${selectedInsightConversationId === row.conversation_id ? 'active' : ''}`}
                    onClick={() => setSelectedInsightConversationId(row.conversation_id)}
                  >
                    <span>{row.timestamp?.slice(0, 19) || 'N/A'}</span>
                    <span>{row.financial_topic || 'General'}</span>
                    <span>{row.risk_level || 'LOW'}</span>
                    <span>{Math.round((row.confidence_score || 0) * 100)}%</span>
                  </button>
                ))}
              </div>

              {selectedInsightConversation && (
                <div className="drilldown-details">
                  <h4>{selectedInsightConversation.financial_topic}</h4>
                  <p><strong>Risk:</strong> {selectedInsightConversation.risk_level}</p>
                  <p><strong>Confidence:</strong> {Math.round((selectedInsightConversation.confidence_score || 0) * 100)}%</p>
                  <p><strong>Summary:</strong> {selectedInsightConversation.executive_summary}</p>
                  <p><strong>Transcript:</strong> {selectedInsightConversation.transcript || 'No transcript available.'}</p>
                  <p><strong>Sentiment:</strong> {selectedInsightConversation.financial_sentiment || 'N/A'}</p>
                  <p><strong>Pipeline Latency:</strong> {(selectedInsightConversation.timing?.total_s || 0).toFixed(2)}s</p>
                  <p><strong>Thread:</strong> {selectedInsightConversation.chat_thread_id || 'N/A'}</p>
                  <p><strong>Details:</strong> Open this thread in Chat to see full Qwen reasoning and transcript editor.</p>
                  <button
                    className="drilldown-open"
                    onClick={() => {
                      const tid = selectedInsightConversation.chat_thread_id || '';
                      if (tid) void openHistoryThread(tid);
                    }}
                  >
                    Open Full Details In Chat
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {view === 'settings' && (
          <div className="insights-wrap">
            <h2>Settings & Legal</h2>
            <div className="settings-grid">
              <div className="insight-card">
                <h3>Appearance</h3>
                <label>Theme Mode</label>
                <select value={themeMode} onChange={(e) => setThemeMode(e.target.value as 'auto' | 'dark' | 'light')}>
                  <option value="auto">Auto (System)</option>
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                </select>
                <label>Font Size ({fontSizePx}px)</label>
                <input type="range" min={12} max={18} value={fontSizePx} onChange={(e) => setFontSizePx(Number(e.target.value))} />
                <label>Text Color</label>
                <input type="color" value={textColor} onChange={(e) => setTextColor(e.target.value)} />
              </div>
              <div className="insight-card">
                <h3>Audio Preferences</h3>
                <label>Transcription Language</label>
                <select value={audioLanguage} onChange={(e) => setAudioLanguage(e.target.value as 'auto' | 'hi' | 'en')}>
                  <option value="auto">Auto</option>
                  <option value="hi">Hindi</option>
                  <option value="en">English</option>
                </select>
                <label>TTS Voice Profile</label>
                <select value={ttsVoiceProfile} onChange={(e) => setTtsVoiceProfile(e.target.value as 'auto' | 'female' | 'male')}>
                  <option value="auto">Auto</option>
                  <option value="female">Female (preferred)</option>
                  <option value="male">Male (preferred)</option>
                </select>
                <p style={{ marginTop: 10, color: 'var(--text-secondary)', fontSize: 13 }}>
                  TTS uses browser voices with Hindi/English auto-detect and short advisor-style insight playback.
                </p>
              </div>
              <div className="insight-card">
                <h3><Shield size={14} style={{ marginRight: 6 }} /> Legal</h3>
                <p><strong>Privacy Policy:</strong> We store encrypted audio and masked transcripts for intelligence extraction.</p>
                <p><strong>Terms:</strong> FinFlux provides structured analysis, not financial advice.</p>
                <p><strong>Data Usage:</strong> Conversation analytics are scoped to authenticated user context.</p>
                <p><strong>Retention:</strong> Users can purge history from the dashboard.</p>
              </div>
            </div>
          </div>
        )}
        </main>
      </div>
    </SidebarProvider>
  );
}

export default App;
