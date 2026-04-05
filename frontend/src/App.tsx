import { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, BarChart3, ChevronLeft, ChevronRight, Loader2, LogOut, MessageSquare, Mic, Pause, Play, Plus, Send, Settings, Shield, UserRound } from 'lucide-react';

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
  const [token, setToken] = useState<string>('');
  const [username, setUsername] = useState<string>('');
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [authUser, setAuthUser] = useState('');
  const [authPass, setAuthPass] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const [view, setView] = useState<ViewMode>('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isRecordPaused, setIsRecordPaused] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingLevels, setRecordingLevels] = useState<number[]>(Array.from({ length: 28 }, () => 0.06));
  const [error, setError] = useState('');
  const [voiceReplyOn] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [editingConversationId, setEditingConversationId] = useState('');
  const [editTranscript, setEditTranscript] = useState('');
  const [isSavingEdit, setIsSavingEdit] = useState(false);
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
      if (!activeThreadId && rows.length > 0) {
        await openHistoryThread(rows[0].thread_id);
      }
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    loadHistory();
    loadThreads();
  }, [token]);

  useEffect(() => {
    const root = document.documentElement;
    const applyTheme = () => {
      if (themeMode === 'auto') {
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        root.setAttribute('data-theme', isDark ? 'dark' : 'light');
      } else {
        root.setAttribute('data-theme', themeMode);
      }
    };
    applyTheme();
    if (themeMode === 'auto' && window.matchMedia) {
      const mql = window.matchMedia('(prefers-color-scheme: dark)');
      const listener = () => applyTheme();
      mql.addEventListener('change', listener);
      return () => mql.removeEventListener('change', listener);
    }
    return undefined;
  }, [themeMode]);

  useEffect(() => {
    document.documentElement.style.setProperty('--app-font-size', `${fontSizePx}px`);
  }, [fontSizePx]);

  useEffect(() => {
    document.documentElement.style.setProperty('--app-text-color', textColor);
  }, [textColor]);

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
      const rows = (data.results || []) as Array<{ id: string; role: 'user' | 'assistant'; text: string; attached_result?: AnalysisResult }>;
      setMessages(rows.map((r) => ({
        id: r.id,
        role: r.role,
        text: r.text,
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

  const toggleExpanded = (messageId: string) => {
    setExpanded((prev) => ({ ...prev, [messageId]: !prev[messageId] }));
  };

  const beginTranscriptEdit = (conversationId: string, transcript: string) => {
    setEditingConversationId(conversationId);
    setEditTranscript(transcript);
  };

  const cancelTranscriptEdit = () => {
    setEditingConversationId('');
    setEditTranscript('');
  };

  const saveTranscriptEdit = async (conversationId: string) => {
    if (!editTranscript.trim()) {
      setError('Transcript cannot be empty.');
      return;
    }
    setIsSavingEdit(true);
    setError('');
    try {
      const res = await authFetch(`/api/conversations/${conversationId}/transcript`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: editTranscript, reanalyze: true }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Transcript update failed' }));
        throw new Error(body.detail || 'Transcript update failed');
      }
      cancelTranscriptEdit();
      await loadHistory();
      if (activeThreadId) await openHistoryThread(activeThreadId);
      setSelectedDetail((prev) => {
        if (!prev || prev.conversation_id !== conversationId) return prev;
        return { ...prev, transcript: editTranscript };
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSavingEdit(false);
    }
  };

  const handleAuth = async () => {
    setAuthLoading(true);
    setAuthError('');
    try {
      const endpoint = authMode === 'signin' ? '/api/auth/login' : '/api/auth/signup';
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: authUser, password: authPass }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Auth failed' }));
        throw new Error(err.detail || 'Auth failed');
      }
      const data = await res.json();
      if (!data.access_token) {
        setAuthError(data.message || 'Signup successful. Please verify your email, then sign in.');
        setAuthMode('signin');
        setAuthPass('');
        return;
      }
      setToken(data.access_token);
      setUsername(data.username);
      setAuthPass('');
      setAuthUser('');
      setView('chat');
    } catch (e) {
      setAuthError(String(e));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleAuthKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      void handleAuth();
    }
  };

  const handleSignOut = () => {
    setToken('');
    setUsername('');
    setHistory([]);
    setThreads([]);
    setMessages([]);
    setActiveThreadId('');
  };

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
      void sendText();
    }
  };

  const downloadPdfReport = async (conversationId: string) => {
    try {
      const res = await authFetch(`/api/report/${conversationId}?format=pdf`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'PDF export failed' }));
        throw new Error(body.detail || 'PDF export failed');
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `finflux_${conversationId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    }
  };

  const sendText = async () => {
    const text = input.trim();
    if (!text || !token || isSending) return;

    const userMessage: ChatMessage = { id: `u-${Date.now()}`, role: 'user', text };
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
      };
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
    const userMessage: ChatMessage = { id: `u-${Date.now()}`, role: 'user', text: 'Voice message sent' };
    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

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
        }]);
        await streamAssistantMessage(assistantId, assistantText, data);
        setSelectedDetail(data);
      } else {
        setMessages((prev) => [...prev, {
          id: assistantId,
          role: 'assistant',
          text: assistantText,
        }]);
      }
      setActiveThreadId(data.chat_thread_id || activeThreadId);
      if (voiceReplyOn) speak(assistantText, data);
      await loadHistory();
      await loadThreads();
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSending(false);
      setRecordingSeconds(0);
    }
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

  if (!token) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <div className="auth-brand"><Activity size={20} /> FinFlux</div>
          <h1>{authMode === 'signin' ? 'Sign In' : 'Create Account'}</h1>
          <p>Secure financial intelligence workspace</p>

          <input
            placeholder="Email"
            value={authUser}
            onChange={(e) => setAuthUser(e.target.value)}
            onKeyDown={handleAuthKeyDown}
          />
          <input
            type="password"
            placeholder="Password"
            value={authPass}
            onChange={(e) => setAuthPass(e.target.value)}
            onKeyDown={handleAuthKeyDown}
          />

          {authError && <div className="auth-error">{authError}</div>}

          <button className="auth-btn" onClick={handleAuth} disabled={authLoading}>
            {authLoading ? <Loader2 className="spin" size={16} /> : authMode === 'signin' ? 'Sign In' : 'Sign Up'}
          </button>

          <button className="auth-switch" onClick={() => setAuthMode((m) => (m === 'signin' ? 'signup' : 'signin'))}>
            {authMode === 'signin' ? 'No account? Create one' : 'Already have an account? Sign in'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`chat-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <aside className="chat-sidebar">
        <div className="sidebar-top">
          <div className="brand-row">
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}><Activity size={16} /> {!sidebarCollapsed ? 'FinFlux' : ''}</span>
            <button className="collapse-btn" onClick={() => setSidebarCollapsed((v) => !v)}>
              {sidebarCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
            </button>
          </div>
          <button className="new-chat-btn" onClick={createNewChat}><Plus size={14} /> New Chat</button>
          <button className={`side-tab ${view === 'chat' ? 'active' : ''}`} onClick={() => setView('chat')}><MessageSquare size={14} /> Current Chat</button>
          <button className={`side-tab ${view === 'insights' ? 'active' : ''}`} onClick={() => setView('insights')}><BarChart3 size={14} /> Insights + Investments</button>
          <button className={`side-tab ${view === 'settings' ? 'active' : ''}`} onClick={() => setView('settings')}><Settings size={14} /> Settings</button>
        </div>

        <div className="history-list">
          {threads.map((item) => (
            <button
              key={item.thread_id}
              className={`history-item ${activeThreadId === item.thread_id ? 'active' : ''}`}
              onClick={() => openHistoryThread(item.thread_id)}
            >
              <div className="history-topic">History · {item.topic || 'General'} · {item.count}</div>
              {!sidebarCollapsed && <div className="history-snippet">{(item.preview || '').slice(0, 70)}</div>}
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-chip"><UserRound size={14} /> {!sidebarCollapsed ? username : ''}</div>
          <button className="signout-btn" onClick={handleSignOut}><LogOut size={14} /> Sign out</button>
        </div>
      </aside>

      <main className="chat-main">
        {view === 'chat' && (
          <div className="chat-workspace">
            <div className="chat-left-pane">
              <div className="messages-wrap">
              {messages.length === 0 && (
                <div className="empty-state">
                  <h2>How can I help with your financial call intelligence?</h2>
                  <p>Type your query to generate transcript-based risk insights and summaries.</p>
                </div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>
                  <div className="msg-bubble" onClick={() => msg.attachedResult && setSelectedDetail(msg.attachedResult)}>
                    <p>{msg.text}</p>
                    {msg.attachedResult?.response_mode === 'analysis' && (
                      <>
                        <div className="msg-metrics">
                          <span>Topic: {msg.attachedResult.financial_topic}</span>
                          <span>Risk: {msg.attachedResult.risk_level}</span>
                          <span>Confidence: {Math.round((msg.attachedResult.confidence_score || 0) * 100)}%</span>
                          <button className="details-toggle" onClick={() => toggleExpanded(msg.id)}>
                            {expanded[msg.id] ? 'Hide details' : 'Show details'}
                          </button>
                        </div>

                        <div className="quick-transcript">
                          <div className="quick-transcript-title">
                            <span>Transcript</span>
                            {editingConversationId !== msg.attachedResult.conversation_id && (
                              <button
                                className="details-toggle"
                                onClick={() => beginTranscriptEdit(msg.attachedResult!.conversation_id, msg.attachedResult!.transcript || '')}
                              >
                                Edit + Regenerate Summary
                              </button>
                            )}
                          </div>
                          {editingConversationId === msg.attachedResult.conversation_id ? (
                            <>
                              <textarea
                                className="transcript-editor"
                                value={editTranscript}
                                onChange={(e) => setEditTranscript(e.target.value)}
                                rows={7}
                              />
                              <div className="transcript-actions">
                                <button onClick={() => saveTranscriptEdit(msg.attachedResult!.conversation_id)} disabled={isSavingEdit}>
                                  {isSavingEdit ? 'Saving...' : 'Save + Regenerate Summary'}
                                </button>
                                <button onClick={cancelTranscriptEdit}>Cancel</button>
                              </div>
                            </>
                          ) : (
                            <p className="quick-transcript-text">{msg.attachedResult.transcript || 'No transcript available.'}</p>
                          )}
                        </div>

                        {expanded[msg.id] && (
                          <div className="message-details">
                            <div className="details-grid">
                              <div className="detail-card">
                                <h4>Strategic Intent</h4>
                                <p>{msg.attachedResult.strategic_intent || 'N/A'}</p>
                              </div>
                              <div className="detail-card">
                                <h4>Future Gearing</h4>
                                <p>{msg.attachedResult.future_gearing || 'N/A'}</p>
                              </div>
                              <div className="detail-card">
                                <h4>Risk Assessment</h4>
                                <p>{msg.attachedResult.risk_assessment || 'N/A'}</p>
                              </div>
                              <div className="detail-card">
                                <h4>Pipeline Latency</h4>
                                <p>{(msg.attachedResult.timing?.total_s || 0).toFixed(2)}s</p>
                              </div>
                            </div>

                            <div className="detail-wall">
                              <h4>Qwen Wall of Logic</h4>
                              <div className="wall-content">{msg.attachedResult.expert_reasoning_points || 'No reasoning available.'}</div>
                            </div>
                            <div className="detail-transcript">
                              <div className="transcript-header">
                                <h4>Report</h4>
                                <button onClick={() => downloadPdfReport(msg.attachedResult!.conversation_id)}>Export PDF</button>
                              </div>
                              <p className="transcript-text">PDF includes full transcript, strategic analysis, and entity wall.</p>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
              {error && <div className="error-line">{error}</div>}
              </div>
            </div>

            <div className="composer">
              <div className="audio-first-panel">
                <div className="audio-panel-head">
                  <div>
                    <div className="audio-title">Audio Recorder</div>
                    <div className="audio-subtitle">Record in Hindi/English with live spikes, then send for analysis</div>
                  </div>
                  <div className="recording-time">{`${Math.floor(recordingSeconds / 60).toString().padStart(2, '0')}:${(recordingSeconds % 60).toString().padStart(2, '0')}`}</div>
                </div>

                <div className="spike-row" aria-hidden="true">
                  {recordingLevels.map((lvl, i) => (
                    <span key={i} className={`spike ${isRecording ? 'active' : ''}`} style={{ height: `${Math.max(10, Math.round(56 * lvl))}px` }} />
                  ))}
                </div>

                <div className="audio-panel-actions">
                  {!isRecording ? (
                    <button className="audio-btn" onClick={startRecord} disabled={isSending}>
                      <Mic size={16} /> Start
                    </button>
                  ) : (
                    <>
                      <button className="audio-btn" onClick={togglePauseRecord}>
                        {isRecordPaused ? <Play size={16} /> : <Pause size={16} />} {isRecordPaused ? 'Resume' : 'Pause'}
                      </button>
                      <button className="audio-btn recording" onClick={stopRecord}>
                        <Mic size={16} /> Send
                      </button>
                      <button className="audio-cancel-btn" onClick={cancelRecord}>Cancel</button>
                    </>
                  )}
                  <select value={audioLanguage} onChange={(e) => setAudioLanguage(e.target.value as 'auto' | 'hi' | 'en')} className="audio-lang-select">
                    <option value="auto">Audio Language: Auto</option>
                    <option value="hi">Audio Language: Hindi</option>
                    <option value="en">Audio Language: English</option>
                  </select>
                </div>
              </div>

              <textarea
                placeholder="Ask FinFlux anything about this conversation..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onComposerKeyDown}
                rows={2}
              />
              <div className="composer-actions">
                <button className="send-btn" onClick={sendText} disabled={isSending || !input.trim()}>
                  {isSending ? <Loader2 className="spin" size={16} /> : <Send size={16} />} Send
                </button>
              </div>
            </div>

            <aside className="chat-right-pane">
              <h3>Analysis Details</h3>
              {!selectedDetail && <p className="detail-placeholder">Select an analysis response to view full risk, entities, and strategic context.</p>}
              {selectedDetail && (
                <div className="right-details-card">
                  <div className="quick-transcript right-transcript">
                    <div className="quick-transcript-title">
                      <span>Full Transcript</span>
                      {editingConversationId !== selectedDetail.conversation_id && (
                        <button
                          className="details-toggle"
                          onClick={() => beginTranscriptEdit(selectedDetail.conversation_id, selectedDetail.transcript || '')}
                        >
                          Edit + Regenerate Summary
                        </button>
                      )}
                    </div>
                    {editingConversationId === selectedDetail.conversation_id ? (
                      <>
                        <textarea
                          className="transcript-editor"
                          value={editTranscript}
                          onChange={(e) => setEditTranscript(e.target.value)}
                          rows={8}
                        />
                        <div className="transcript-actions">
                          <button onClick={() => saveTranscriptEdit(selectedDetail.conversation_id)} disabled={isSavingEdit}>
                            {isSavingEdit ? 'Saving...' : 'Save + Regenerate Summary'}
                          </button>
                          <button onClick={cancelTranscriptEdit}>Cancel</button>
                        </div>
                      </>
                    ) : (
                      <p className="quick-transcript-text">{selectedDetail.transcript || 'No transcript available.'}</p>
                    )}
                  </div>

                  <p><strong>Summary:</strong> {selectedDetail.executive_summary || 'N/A'}</p>
                  <p><strong>Topic:</strong> {selectedDetail.financial_topic || 'N/A'}</p>
                  <p><strong>Risk:</strong> {selectedDetail.risk_level || 'LOW'}</p>
                  <p><strong>Confidence:</strong> {Math.round((selectedDetail.confidence_score || 0) * 100)}%</p>
                  <p><strong>Sentiment:</strong> {selectedDetail.financial_sentiment || 'Neutral'}</p>
                  <p><strong>Language:</strong> {String(selectedDetail.language || 'unknown').toUpperCase()}</p>
                  <p><strong>Strategic Intent:</strong> {selectedDetail.strategic_intent || 'N/A'}</p>
                  <p><strong>Future Gearing:</strong> {selectedDetail.future_gearing || 'N/A'}</p>
                  <p><strong>Risk Assessment:</strong> {selectedDetail.risk_assessment || 'N/A'}</p>

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
          </div>
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
  );
}

export default App;
