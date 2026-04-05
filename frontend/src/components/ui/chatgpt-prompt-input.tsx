import { useMemo, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { Mic, Paperclip, Pause, Play, Send, Upload, X } from 'lucide-react';

type PromptBoxProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  isSending: boolean;
  isRecording: boolean;
  isRecordPaused: boolean;
  recordingLevels: number[];
  recordingSeconds: number;
  audioLanguage: 'auto' | 'hi' | 'en';
  onAudioLanguageChange: (value: 'auto' | 'hi' | 'en') => void;
  onStartRecord: () => void;
  onTogglePauseRecord: () => void;
  onStopRecord: () => void;
  onCancelRecord: () => void;
  files: File[];
  onPickFiles: (files: File[]) => void;
  onRemoveFile: (index: number) => void;
};

export function PromptBox({
  value,
  onChange,
  onSubmit,
  onKeyDown,
  isSending,
  isRecording,
  isRecordPaused,
  recordingLevels,
  recordingSeconds,
  audioLanguage,
  onAudioLanguageChange,
  onStartRecord,
  onTogglePauseRecord,
  onStopRecord,
  onCancelRecord,
  files,
  onPickFiles,
  onRemoveFile,
}: PromptBoxProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const timeLabel = useMemo(() => {
    const mins = Math.floor(recordingSeconds / 60).toString().padStart(2, '0');
    const secs = (recordingSeconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  }, [recordingSeconds]);

  const resizeTextarea = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  };

  const onTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(event.target.value);
    requestAnimationFrame(resizeTextarea);
  };

  const handleFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    onPickFiles(Array.from(list));
  };

  return (
    <form
      className={`prompt-box ${dragOver ? 'drag-over' : ''}`}
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragOver(false);
        handleFiles(event.dataTransfer.files);
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        className="prompt-file-input"
        multiple
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = '';
        }}
      />

      {files.length > 0 && (
        <div className="prompt-files-row">
          {files.map((file, index) => (
            <div className="prompt-file-chip" key={`${file.name}-${file.size}-${index}`}>
              <Upload size={14} />
              <span>{file.name}</span>
              <button type="button" onClick={() => onRemoveFile(index)} aria-label={`Remove ${file.name}`}>
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={onTextareaChange}
        onKeyDown={onKeyDown}
        placeholder="Upload or record a financial call, then ask for risk and advisory insights..."
        className="prompt-textarea"
      />

      {isRecording && (
        <div className="prompt-recorder-panel">
          <div className="prompt-recorder-head">
            <span>Listening {isRecordPaused ? '(paused)' : ''}</span>
            <span>{timeLabel}</span>
          </div>
          <div className="prompt-wave-row" aria-hidden="true">
            {recordingLevels.map((level, index) => (
              <span
                key={index}
                className={`prompt-wave ${isRecordPaused ? 'paused' : 'active'}`}
                style={{ height: `${Math.max(8, Math.round(42 * level))}px` }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="prompt-actions-row">
        <div className="prompt-actions-left">
          <button
            type="button"
            className="prompt-icon-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Upload call audio"
          >
            <Paperclip size={16} />
          </button>

          {!isRecording ? (
            <button
              type="button"
              className="prompt-icon-btn prompt-mic-btn"
              onClick={onStartRecord}
              disabled={isSending}
              title="Start call recording"
            >
              <Mic size={16} />
            </button>
          ) : (
            <>
              <button type="button" className="prompt-mini-btn" onClick={onTogglePauseRecord}>
                {isRecordPaused ? <Play size={14} /> : <Pause size={14} />}
                {isRecordPaused ? 'Resume' : 'Pause'}
              </button>
              <button type="button" className="prompt-mini-btn recording" onClick={onStopRecord}>
                <Mic size={14} />
                Send Call Audio
              </button>
              <button type="button" className="prompt-mini-btn cancel" onClick={onCancelRecord}>
                <X size={14} />
                Cancel
              </button>
            </>
          )}

          <select
            value={audioLanguage}
            className="prompt-audio-lang"
            onChange={(event) => onAudioLanguageChange(event.target.value as 'auto' | 'hi' | 'en')}
          >
            <option value="auto">Audio: Auto</option>
            <option value="hi">Audio: Hindi</option>
            <option value="en">Audio: English</option>
          </select>
        </div>

        <button type="submit" className="prompt-send-btn" disabled={isSending} title="Analyze input">
          <Send size={15} />
          {isSending ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
    </form>
  );
}
