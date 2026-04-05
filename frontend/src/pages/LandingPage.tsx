import { useEffect, useState } from 'react';
import { GridScan } from '../components/GridScan.jsx';
import DecryptedText from '../components/DecryptedText.jsx';

type LandingPageProps = {
  onGetStarted: () => void;
};

const highlights = [
  { label: 'Strategic Risk Synthesis', desc: 'Enhanced: Powered by Advanced AI: Leverage multi-layered LLMs (Qwen & Llama) to instantly map structured intent, forecast future gearing, and output deep risk assessments.' },
  { label: 'Intelligence Dashboard', desc: 'Enhanced: Command Your Data: Monitor risk distributions, spot emerging trend lines, and isolate high-stakes conversations—all from a single, intuitive pane of glass.' },
  { label: 'Secure Audio Ingestion', desc: 'Store captured audio with AES-256 encryption and controlled persistence.' },
  { label: 'Multilingual Financial ASR', desc: 'Whisper Turbo pipeline handles financial speech with support for multilingual conversations.' },
  { label: 'Expert NLP Stack', desc: 'FinBERT, GLiNER, and DeBERTa models extract sentiment, entities, topics, and obligations.' },
  { label: 'Report Export', desc: 'Enhanced: Seamless Governance: Instantly generate and download compliance-ready PDF and CSV reports for effortless team analysis and operational oversight.' },
];

export default function LandingPage({ onGetStarted }: LandingPageProps) {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 600);
    const t2 = setTimeout(() => setPhase(2), 1200);
    const t3 = setTimeout(() => setPhase(3), 2000);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, []);

  return (
    <div className="landing-page" style={{ position: 'relative', minHeight: '100vh', background: '#030506', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {/* GridScan Background */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <GridScan
          sensitivity={0.55}
          lineThickness={1}
          linesColor="#0a2a1a"
          scanColor="#00ff9c"
          scanOpacity={0.4}
          gridScale={0.1}
          lineStyle="solid"
          lineJitter={0.1}
          scanDirection="pingpong"
          noiseIntensity={0.01}
          scanGlow={0.5}
          scanSoftness={2}
          scanDuration={2}
          scanDelay={2}
          scanOnClick={false}
          enablePost={false}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      {/* Content Overlay */}
      <div
        style={{
          position: 'relative',
          zIndex: 1,
          maxWidth: '820px',
          margin: '0 auto',
          padding: '0 24px',
          textAlign: 'center',
        }}
      >
        {/* Tag (Eyebrow) */}
        <div
          style={{
            opacity: phase >= 1 ? 1 : 0,
            transform: phase >= 1 ? 'translateY(0)' : 'translateY(12px)',
            transition: 'opacity 600ms ease, transform 600ms ease',
            marginBottom: '18px',
          }}
        >
          <span className="landing-tag">AUDIO-FIRST FINANCIAL ASSISTANCE</span>
        </div>

        {/* Headline */}
        <h1
          className="landing-title"
          style={{
            opacity: phase >= 2 ? 1 : 0,
            transition: 'opacity 400ms ease',
            lineHeight: 1.1,
            marginBottom: '24px',
            fontSize: '5.5rem',
            letterSpacing: '-0.02em',
            background: 'linear-gradient(180deg, #fff 0%, #00ff9c 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            display: 'inline-block'
          }}
        >
          {phase >= 2 ? (
            <DecryptedText
              text="FinFlux"
              speed={45}
              maxIterations={12}
              characters="01!@#%"
              animateOn="view"
              revealDirection="center"
              sequential
              className="landing-char-revealed"
              encryptedClassName="landing-char-encrypted"
            />
          ) : (
            <span className="landing-char-encrypted">finflux</span>
          )}
        </h1>

        {/* Sub-headline */}
        <p
          className="landing-subtitle"
          style={{
            opacity: phase >= 2 ? 1 : 0,
            transform: phase >= 2 ? 'translateY(0)' : 'translateY(8px)',
            transition: 'opacity 500ms ease 300ms, transform 500ms ease 300ms',
            maxWidth: '800px',
            margin: '0 auto 40px auto',
            fontSize: '1.4rem',
            lineHeight: 1.5,
            color: '#e2e8f0',
            fontWeight: 500
          }}
        >
          Transform Conversations into Actionable Financial Intelligence.
        </p>

        {/* Feature grid */}
        <div className="landing-grid" style={{ opacity: phase >= 1 ? 1 : 0, transform: phase >= 1 ? 'translateY(0)' : 'translateY(18px)', transition: 'opacity 700ms ease 100ms, transform 700ms ease 100ms' }}>
          {highlights.map((h, i) => (
            <div
              key={h.label}
              className="landing-card electric-border"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <h3>{h.label}</h3>
              <p>{h.desc}</p>
            </div>
          ))}
        </div>

        {/* Section 3 CTA */}
        <div
          style={{
            opacity: phase >= 3 ? 1 : 0,
            transform: phase >= 3 ? 'translateY(0)' : 'translateY(10px)',
            transition: 'opacity 500ms ease, transform 500ms ease',
            marginTop: '48px',
          }}
        >
          <h2 style={{ fontSize: '24px', letterSpacing: '0.02em', marginBottom: '24px', color: '#fff' }}>Turn Every Conversation into a Strategic Asset.</h2>
          <button className="landing-cta" onClick={onGetStarted}>
            Try FinFlux Now
            <span className="landing-cursor">|</span>
          </button>

          <p style={{ marginTop: '14px', fontSize: '0.72rem', color: '#3a5a4a', letterSpacing: '0.04em' }}>
            Free to use · No financial advice · Your data stays encrypted
          </p>
        </div>
      </div>
    </div>
  );
}
