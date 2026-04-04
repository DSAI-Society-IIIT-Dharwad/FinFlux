import { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, BarChart3, Loader2, LogOut, MessageSquare, Mic, Plus, Send, UserRound, Volume2 } from 'lucide-react';

type ViewMode = 'chat' | 'insights';

interface TopicScore { topic: string; score: number }
interface SentimentBreakdown { positive?: number; neutral?: number; negative?: number }
interface Entity { type: string; value: string; confidence?: number }
interface AnalysisResult {
  conversation_id: string;
  chat_thread_id?: string;
  timestamp: string;
  financial_topic: string;
  risk_level: string;
  financial_sentiment: string;
  confidence_score: number;
  executive_summary: string;
  transcript: string;
  strategic_intent?: string;
  future_gearing?: string;
  risk_assessment?: string;
  expert_reasoning_points?: string;
  timing?: { total_s?: number };
  topic_top3?: TopicScore[];
  sentiment_breakdown?: SentimentBreakdown;
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
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingLevels, setRecordingLevels] = useState<number[]>(Array.from({ length: 28 }, () => 0.06));
  const [error, setError] = useState('');
  const [voiceReplyOn, setVoiceReplyOn] = useState(true);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [editingConversationId, setEditingConversationId] = useState('');
  const [editTranscript, setEditTranscript] = useState('');
  const [isSavingEdit, setIsSavingEdit] = useState(false);
  const [selectedInsightConversationId, setSelectedInsightConversationId] = useState('');

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

  const speak = (text: string) => {
    if (!text.trim() || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
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
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        text: '',
      };
      setMessages((prev) => [...prev, assistantMessage]);
      const textOut = data.executive_summary || 'Analysis complete.';
      await streamAssistantMessage(assistantId, textOut, data);
      if (voiceReplyOn) speak(textOut);
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
        pcmBufferRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
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
      const res = await authFetch('/api/analyze', { method: 'POST', body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Audio analyze failed' }));
        throw new Error(body.detail || 'Audio analyze failed');
      }
      const data: AnalysisResult = await res.json();
      const assistantId = `a-${Date.now()}`;
      setMessages((prev) => [...prev, {
        id: assistantId,
        role: 'assistant',
        text: '',
      }]);
      const textOut = data.executive_summary || 'Audio analysis complete.';
      await streamAssistantMessage(assistantId, textOut, data);
      setActiveThreadId(data.chat_thread_id || activeThreadId);
      if (voiceReplyOn) speak(textOut);
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
    <div className="chat-shell">
      <aside className="chat-sidebar">
        <div className="sidebar-top">
          <div className="brand-row"><Activity size={16} /> FinFlux</div>
          <button className="new-chat-btn" onClick={createNewChat}><Plus size={14} /> New Chat</button>
          <button className={`side-tab ${view === 'chat' ? 'active' : ''}`} onClick={() => setView('chat')}><MessageSquare size={14} /> Chat</button>
          <button className={`side-tab ${view === 'insights' ? 'active' : ''}`} onClick={() => setView('insights')}><BarChart3 size={14} /> Insights</button>
        </div>

        <div className="history-list">
          {threads.map((item) => (
            <button
              key={item.thread_id}
              className={`history-item ${activeThreadId === item.thread_id ? 'active' : ''}`}
              onClick={() => openHistoryThread(item.thread_id)}
            >
              <div className="history-topic">{item.topic || 'General'} · {item.count}</div>
              <div className="history-snippet">{(item.preview || '').slice(0, 70)}</div>
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-chip"><UserRound size={14} /> {username}</div>
          <button className="signout-btn" onClick={handleSignOut}><LogOut size={14} /> Sign out</button>
        </div>
      </aside>

      <main className="chat-main">
        {view === 'chat' && (
          <>
            <div className="messages-wrap">
              {messages.length === 0 && (
                <div className="empty-state">
                  <h2>How can I help with your financial call intelligence?</h2>
                  <p>Type a message or use audio to run full pipeline analysis.</p>
                </div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className={`msg ${msg.role}`}>
                  <div className="msg-bubble">
                    <p>{msg.text}</p>
                    {msg.role === 'assistant' && (
                      <button className="speak-btn" onClick={() => speak(msg.text)}>
                        <Volume2 size={14} /> Play voice
                      </button>
                    )}
                    {msg.attachedResult && (
                      <>
                        <div className="msg-metrics">
                          <span>Topic: {msg.attachedResult.financial_topic}</span>
                          <span>Risk: {msg.attachedResult.risk_level}</span>
                          <span>Confidence: {Math.round((msg.attachedResult.confidence_score || 0) * 100)}%</span>
                          <button
                            className="details-toggle"
                            onClick={() => beginTranscriptEdit(msg.attachedResult!.conversation_id, msg.attachedResult!.transcript || '')}
                          >
                            Edit transcript
                          </button>
                          <button className="details-toggle" onClick={() => toggleExpanded(msg.id)}>
                            {expanded[msg.id] ? 'Hide details' : 'Show details'}
                          </button>
                        </div>

                        <div className="quick-transcript">
                          <div className="quick-transcript-title">Transcript</div>
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
                                  {isSavingEdit ? 'Saving...' : 'Save + Reanalyze'}
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

            <div className="composer">
              <div className="audio-first-panel">
                <div className="audio-panel-head">
                  <div>
                    <div className="audio-title">Audio First Capture</div>
                    <div className="audio-subtitle">Record the call input first, then add optional text below</div>
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
                      <Mic size={16} /> Start Recording
                    </button>
                  ) : (
                    <>
                      <button className="audio-btn recording" onClick={stopRecord}>
                        <Mic size={16} /> Stop & Send
                      </button>
                      <button className="audio-cancel-btn" onClick={cancelRecord}>Cancel</button>
                    </>
                  )}
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
                <button className={`voice-toggle ${voiceReplyOn ? 'on' : ''}`} onClick={() => setVoiceReplyOn((v) => !v)}>
                  <Volume2 size={16} /> Voice reply {voiceReplyOn ? 'On' : 'Off'}
                </button>
                <button className={`audio-btn ${isRecording ? 'recording' : ''}`} onClick={isRecording ? stopRecord : startRecord}>
                  <Mic size={16} /> {isRecording ? 'Stop' : 'Audio'}
                </button>
                <button className="send-btn" onClick={sendText} disabled={isSending || !input.trim()}>
                  {isSending ? <Loader2 className="spin" size={16} /> : <Send size={16} />} Send
                </button>
              </div>
            </div>
          </>
        )}

        {view === 'insights' && (
          <div className="insights-wrap">
            <h2>Insights</h2>
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
      </main>
    </div>
  );
}

export default App;
