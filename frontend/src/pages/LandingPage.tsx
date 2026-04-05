import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import { Sparkles } from 'lucide-react';
import VapourTextEffect from '../components/VapourTextEffect.tsx';
import FloatingLines from '../components/FloatingLines.tsx';

type LandingPageProps = {
  onGetStarted: () => void;
};

const features = [
  {
    title: 'Secure Audio Ingestion',
    description: 'Store captured audio with AES-256 encryption and controlled persistence.',
  },
  {
    title: 'Multilingual Financial ASR',
    description: 'Whisper Turbo pipeline handles financial speech with support for multilingual conversations.',
  },
  {
    title: 'Expert NLP Stack',
    description: 'FinBERT, GLiNER, and DeBERTa models extract sentiment, entities, topics, and obligations.',
  },
  {
    title: 'Strategic Risk Synthesis',
    description: 'Qwen and Llama layers produce structured intent, future gearing, and risk assessment.',
  },
  {
    title: 'Intelligence Dashboard',
    description: 'Track risk distributions, trend lines, timelines, and high-risk conversations in one view.',
  },
  {
    title: 'Report Export',
    description: 'Generate downloadable PDF and CSV reports for analysis, governance, and operations.',
  },
];

const steps = [
  'Capture financial conversations through the FinFlux interface',
  'Run transcription, transcript normalization, and language detection',
  'Extract entities, sentiment, commitments, and financial topic signals',
  'Generate strategic reasoning, risk assessment, and executive summary',
  'View everything in a unified dashboard',
];

const sectionStyle: CSSProperties = {
  padding: '72px 0',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
};

export default function LandingPage({ onGetStarted }: LandingPageProps) {
  const [showIntro, setShowIntro] = useState(true);
  const [showPage, setShowPage] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => setReducedMotion(media.matches);
    apply();
    if (media.addEventListener) {
      media.addEventListener('change', apply);
      return () => media.removeEventListener('change', apply);
    }
    media.addListener(apply);
    return () => media.removeListener(apply);
  }, []);

  useEffect(() => {
    if (showIntro) return;
    const id = window.setTimeout(() => setShowPage(true), 30);
    return () => window.clearTimeout(id);
  }, [showIntro]);

  return (
    <div className="landing-page" style={{ position: 'relative', minHeight: '100vh', color: '#eaf2ff', overflowX: 'hidden' }}>
      {showIntro && <VapourTextEffect text="FINFLUX" onComplete={() => setShowIntro(false)} />}

      {!reducedMotion && (
        <FloatingLines
          enabledWaves={['top', 'middle', 'bottom']}
          lineCount={3}
          lineDistance={4}
          bendRadius={3.5}
          bendStrength={-0.35}
          interactive={false}
          parallax={false}
          animationSpeed={0.8}
          linesGradient={['#071325', '#1a4410', '#4cbb17', '#9be564']}
          mixBlendMode="screen"
        />
      )}

      <div
        style={{
          position: 'relative',
          zIndex: 1,
          maxWidth: '1160px',
          margin: '0 auto',
          padding: '0 20px 64px',
          textShadow: '0 1px 2px rgba(0, 0, 0, 0.72)',
          opacity: showPage ? 1 : 0,
          transform: showPage ? 'translateY(0)' : 'translateY(16px)',
          transition: 'opacity 420ms ease, transform 420ms ease',
        }}
      >
        <section style={{ ...sectionStyle, paddingTop: '96px' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '18px', color: '#fcd34d', fontSize: '0.88rem', letterSpacing: '0.11em', fontWeight: 700 }}>
            <Sparkles size={14} /> AUDIO-FIRST FINANCIAL ASSISTANCE
          </div>
          <h1 style={{ fontSize: 'clamp(2.6rem, 7vw, 5rem)', lineHeight: 1.05, letterSpacing: '-0.045em', marginBottom: '18px' }}>
            Turn Financial Call Audio into Actionable Financial Guidance
          </h1>
          <div className="glass-panel" style={{ maxWidth: '980px', padding: '24px', background: 'rgba(8,26,30,0.78)' }}>
            <p style={{ fontSize: '1.45rem', lineHeight: 1.72, color: '#dce8f8' }}>
              FinFlux is an audio-native financial aid platform that transforms recorded calls and uploaded voice notes into structured risk and advisory outputs using a secure, multi-stage AI pipeline.
            </p>
            <p style={{ marginTop: '16px', fontSize: '1.26rem', color: '#bfd1e5', lineHeight: 1.7 }}>
              Built for teams that need fast call intelligence, traceability, and consistent financial decision support.
            </p>
          </div>
        </section>

        <section style={sectionStyle}>
          <h2 style={{ fontSize: '2.35rem', marginBottom: '16px', letterSpacing: '-0.035em', color: '#f4f8ff' }}>What is Audio Financial Intelligence?</h2>
          <div className="glass-panel" style={{ maxWidth: '980px', padding: '24px', background: 'rgba(8,26,30,0.78)' }}>
            <p style={{ color: '#dce8f8', lineHeight: 1.85, fontSize: '1.28rem' }}>
              FinFlux combines secure audio ingestion, multilingual ASR, and an expert model stack to analyze financial calls in near real time. It converts unstructured voice communication into strategic signals, including sentiment, risk, entities, intent, and compliance-relevant context.
            </p>
          </div>
        </section>

        <section style={sectionStyle}>
          <h2 style={{ fontSize: '2.35rem', marginBottom: '18px', letterSpacing: '-0.035em', color: '#f4f8ff' }}>Powerful Features Designed for Financial Intelligence</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
            {features.map((feature) => (
              <div key={feature.title} className="glass-panel" style={{ padding: '20px', background: 'rgba(8,26,30,0.78)' }}>
                <h3 style={{ marginBottom: '10px', fontSize: '1.22rem' }}>{feature.title}</h3>
                <p style={{ color: '#dce8f8', lineHeight: 1.8, fontSize: '1.2rem' }}>{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section style={sectionStyle}>
          <h2 style={{ fontSize: '2.35rem', marginBottom: '18px', letterSpacing: '-0.035em', color: '#f4f8ff' }}>From Call Audio to Insight in Seconds</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '12px' }}>
            {steps.map((step, idx) => (
              <div key={step} className="glass-panel" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px 18px', background: 'rgba(8,26,30,0.78)' }}>
                <div style={{ width: '28px', height: '28px', borderRadius: '999px', background: 'rgba(245,158,11,0.18)', color: '#fcd34d', display: 'grid', placeItems: 'center', fontWeight: 700 }}>{idx + 1}</div>
                <p style={{ fontSize: '1.22rem', lineHeight: 1.7 }}>{step}</p>
              </div>
            ))}
          </div>
        </section>

        <section style={{ ...sectionStyle, textAlign: 'center' }}>
          <h2 style={{ fontSize: '2.45rem', marginBottom: '14px', letterSpacing: '-0.035em', color: '#f4f8ff' }}>Unlock the Power of Your Financial Call Recordings</h2>
          <button
            onClick={onGetStarted}
            style={{
              border: 'none',
              borderRadius: '12px',
              padding: '13px 26px',
              fontWeight: 700,
              fontSize: '0.95rem',
              background: 'linear-gradient(135deg, #f59e0b, #f97316)',
              color: '#0b0f1a',
              cursor: 'pointer',
              boxShadow: '0 12px 28px -10px rgba(245,158,11,0.7)',
              marginTop: '8px',
            }}
          >
            Start Audio Analysis
          </button>
        </section>

        <footer style={{ paddingTop: '32px', display: 'flex', justifyContent: 'space-between', gap: '14px', flexWrap: 'wrap', color: '#bfd1e5', fontSize: '1rem' }}>
          <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
            <span>About Us</span>
            <span>Contact</span>
            <span>Privacy Policy</span>
            <span>Terms of Service</span>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <span>LinkedIn</span>
            <span>X</span>
            <span>YouTube</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
