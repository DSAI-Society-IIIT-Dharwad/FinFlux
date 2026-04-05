import { useEffect, useMemo, useRef, useState } from 'react';
import { Copy, Mic, Pencil, RotateCcw, Shield } from 'lucide-react';
import LandingPage from './pages/LandingPage.tsx';
import TravelConnectSignin, { type TravelConnectAuthMode } from './components/ui/travel-connect-signin';
import { AppSidebar } from './components/app-sidebar';
import { SidebarProvider, SidebarTrigger } from './components/ui/sidebar';
import { PromptBox } from './components/ui/chatgpt-prompt-input';

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

const API_BASE = 'http://localhost:8000';

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
  const [themeMode, setThemeMode] = useState<'auto' | 'dark' | 'light'>('auto');
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
    const prompt = getPromptByConversationId(conversationId) || fallbackPrompt;
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

  const basicExplainability = useMemo(() => {
    if (!selectedDetail) return null;
    const topEntities = (selectedDetail.entities || []).slice(0, 3).map((e) => `${e.type}: ${e.value}`);
    const topicScores = (selectedDetail.topic_top3 || []).slice(0, 3).map((t) => `${t.topic} (${Math.round(t.score * 100)}%)`);
    const sentiment = selectedDetail.sentiment_breakdown || {};
    const quality = selectedDetail.quality_metrics || {};

    return {
      whatUserSaid: firstSentence(selectedDetail.transcript || 'No transcript available.'),
      whyRisk: firstSentence(selectedDetail.risk_assessment || `Risk classified as ${selectedDetail.risk_level || 'LOW'} from detected obligations and intent signals.`),
      evidence: topEntities.length > 0 ? topEntities : ['No strong entities extracted in this turn.'],
      topicCandidates: topicScores.length > 0 ? topicScores : ['No ranked topic candidates available.'],
      sentimentSummary: `Positive ${Math.round((sentiment.positive || 0) * 100)}%, Neutral ${Math.round((sentiment.neutral || 0) * 100)}%, Negative ${Math.round((sentiment.negative || 0) * 100)}%`,
      nextWatch: firstSentence(selectedDetail.future_gearing || 'Track affordability and commitment follow-through in the next conversation.'),
      languageLine: `${String(selectedDetail.language || 'unknown').toUpperCase()} (${Math.round((selectedDetail.language_confidence || quality.language_confidence || 0) * 100)}% confidence)`,
    };
  }, [selectedDetail]);

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
      const assistantText = (data.response_mode && data.response_mode !== 'analysis')
        ? (data.assistant_text || data.executive_summary || 'Done.')
        : (data.executive_summary || 'Analysis complete.');
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        text: '',
        conversationId: data.conversation_id,
      };
      setMessages((prev) => prev.map((m) => (m.id === userMessageId ? { ...m, conversationId: data.conversation_id } : m)));
      if (data.response_mode === 'analysis') {
        setMessages((prev) => [...prev, assistantMessage]);
        await streamAssistantMessage(assistantId, assistantText, data);
        setSelectedDetail(data);
      } else {
        setMessages((prev) => [...prev, { ...assistantMessage, text: assistantText }]);
      }
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
      const assistantText = (data.response_mode && data.response_mode !== 'analysis')
        ? (data.assistant_text || data.executive_summary || 'Done.')
        : (data.executive_summary || 'Audio analysis complete.');
      if (data.response_mode === 'analysis') {
        setMessages((prev) => [...prev, {
          id: assistantId,
          role: 'assistant',
          text: '',
          conversationId: data.conversation_id,
        }]);
        setMessages((prev) => prev.map((m) => (m.id === userMessageId ? { ...m, conversationId: data.conversation_id } : m)));
        await streamAssistantMessage(assistantId, assistantText, data);
        setSelectedDetail(data);
      } else {
        setMessages((prev) => [...prev, {
          id: assistantId,
          role: 'assistant',
          text: assistantText,
          conversationId: data.conversation_id,
        }]);
        setMessages((prev) => prev.map((m) => (m.id === userMessageId ? { ...m, conversationId: data.conversation_id } : m)));
      }
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
                  <h2>Record or upload a financial call to begin analysis.</h2>
                  <p>FinFlux will transcribe call audio and generate risk, intent, and advisory insights.</p>
                  <div className="empty-audio-cta">
                    <button
                      type="button"
                      className="prompt-icon-btn prompt-mic-btn empty-mic-btn"
                      onClick={startRecord}
                      disabled={isSending || isRecording}
                      title="Start first call recording"
                      aria-label="Start first call recording"
                    >
                      <Mic size={100} />
                    </button>
                  </div>
                </div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>
                  <div className="msg-bubble" onClick={() => msg.attachedResult && setSelectedDetail(msg.attachedResult)}>
                    {msg.role === 'assistant' && <p className="msg-label">Call Summary</p>}
                    <p>{msg.text}</p>
                    {msg.role === 'user' && (
                      <div className="msg-tools" onClick={(e) => e.stopPropagation()}>
                        <button className="msg-tool-btn" onClick={() => void copyPrompt(msg.id, msg.text)} title="Copy prompt" aria-label="Copy prompt">
                          <Copy size={13} />
                          <span>{copiedMessageId === msg.id ? 'Copied' : 'Copy'}</span>
                        </button>
                        <button className="msg-tool-btn" onClick={() => editPrompt(msg.text, msg.conversationId)} title="Edit prompt" aria-label="Edit prompt">
                          <Pencil size={13} />
                          <span>Edit</span>
                        </button>
                      </div>
                    )}
                    {msg.role === 'assistant' && msg.attachedResult?.response_mode === 'analysis' && (
                      <>
                        <div className="msg-meta-line">
                          <span>Topic: {msg.attachedResult.financial_topic}</span>
                          <span>Risk: {msg.attachedResult.risk_level}</span>
                        </div>
                        <div className="summary-actions" onClick={(e) => e.stopPropagation()}>
                          <button
                            className="msg-tool-btn"
                            onClick={() => startConversationEdit(msg.attachedResult!.conversation_id, msg.attachedResult!.transcript || '')}
                            title="Re-run analysis"
                            aria-label="Re-run analysis"
                          >
                            <RotateCcw size={13} />
                            <span>Re-run Analysis</span>
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}
              {error && <div className="error-line">{error}</div>}
                </div>
              </div>

              {messages.length > 0 && (
                <aside className="chat-right-pane">
                  <h3>Call Analysis Details</h3>
                  {!selectedDetail && <p className="detail-placeholder">Select an analysis response to view full risk, entities, and strategic context.</p>}
                  {selectedDetail && (
                    <div className="right-details-card">
                      <div className="analysis-topic-block">
                        <h4>Topic</h4>
                        <p>{selectedDetail.financial_topic || 'N/A'}</p>
                      </div>

                      <div className="analysis-grid">
                        <p><strong>Risk:</strong> {selectedDetail.risk_level || 'LOW'}</p>
                        <p><strong>Confidence:</strong> {Math.round((selectedDetail.confidence_score || 0) * 100)}%</p>
                        <p><strong>Sentiment:</strong> {selectedDetail.financial_sentiment || 'Neutral'}</p>
                        <p><strong>Language:</strong> {String(selectedDetail.language || 'unknown').toUpperCase()}</p>
                        <p><strong>Strategic Intent:</strong> {selectedDetail.strategic_intent || 'N/A'}</p>
                        <p><strong>Future Gearing:</strong> {selectedDetail.future_gearing || 'N/A'}</p>
                        <p><strong>Risk Assessment:</strong> {selectedDetail.risk_assessment || 'N/A'}</p>
                        <p><strong>Pipeline Latency:</strong> {(selectedDetail.timing?.total_s || 0).toFixed(2)}s</p>
                      </div>

                      {basicExplainability && (
                        <div className="explain-box">
                          <h4>Basic Explainability</h4>
                          <p><strong>What user said:</strong> {basicExplainability.whatUserSaid}</p>
                          <p><strong>Why this risk:</strong> {basicExplainability.whyRisk}</p>
                          <p><strong>Language:</strong> {basicExplainability.languageLine}</p>
                          <p><strong>Topic candidates:</strong> {basicExplainability.topicCandidates.join(' | ')}</p>
                          <p><strong>Entity evidence:</strong> {basicExplainability.evidence.join(' | ')}</p>
                          <p><strong>Sentiment signal:</strong> {basicExplainability.sentimentSummary}</p>
                          <p><strong>What to monitor next:</strong> {basicExplainability.nextWatch}</p>
                        </div>
                      )}
                    </div>
                  )}
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
