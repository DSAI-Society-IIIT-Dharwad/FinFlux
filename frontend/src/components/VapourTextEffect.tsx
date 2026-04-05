import { useEffect, useMemo, useRef } from 'react';

type Particle = {
  x: number;
  y: number;
  originX: number;
  originY: number;
  r: number;
  g: number;
  b: number;
  alpha: number;
  baseAlpha: number;
  vx: number;
  vy: number;
};

type TextBounds = {
  left: number;
  right: number;
  width: number;
};

type VapourTextEffectProps = {
  text?: string;
  onComplete?: () => void;
  fadeInMs?: number;
  vaporizeMs?: number;
};

export default function VapourTextEffect({
  text = 'FINFLUX',
  onComplete,
  fadeInMs = 1100,
  vaporizeMs = 2100,
}: VapourTextEffectProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const boundsRef = useRef<TextBounds>({ left: 0, right: 0, width: 0 });
  const rafRef = useRef<number | null>(null);
  const doneRef = useRef(false);

  const totalMs = useMemo(() => fadeInMs + vaporizeMs, [fadeInMs, vaporizeMs]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const fitFontSize = (maxWidthPx: number, baseHeightPx: number) => {
      let size = Math.max(120, Math.floor(baseHeightPx * 0.32));
      ctx.font = `700 ${size}px Outfit, Inter, sans-serif`;
      let measured = ctx.measureText(text).width;
      while (measured > maxWidthPx && size > 60) {
        size -= 4;
        ctx.font = `700 ${size}px Outfit, Inter, sans-serif`;
        measured = ctx.measureText(text).width;
      }
      return size;
    };

    const setup = () => {
      const rect = wrapper.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const fontSize = fitFontSize(canvas.width * 0.9, canvas.height);
      ctx.font = `700 ${fontSize}px Outfit, Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      const measured = ctx.measureText(text).width;
      const extrusionDepth = Math.max(8, Math.min(26, Math.floor(fontSize * 0.14)));

      // Back-to-front stacked layers create the pseudo-3D extrusion.
      for (let i = extrusionDepth; i >= 1; i -= 1) {
        const t = i / extrusionDepth;
        const r = Math.round(6 + (18 - 6) * (1 - t));
        const g = Math.round(20 + (62 - 20) * (1 - t));
        const b = Math.round(45 + (98 - 45) * (1 - t));
        ctx.fillStyle = `rgba(${r},${g},${b},0.94)`;
        ctx.fillText(text, centerX + i * 0.95, centerY + i * 0.68);
      }

      const gradient = ctx.createLinearGradient(centerX - measured / 2, centerY - fontSize * 0.62, centerX + measured / 2, centerY + fontSize * 0.5);
      gradient.addColorStop(0, '#0b1a3d');
      gradient.addColorStop(0.5, '#145071');
      gradient.addColorStop(1, '#facc15');
      ctx.fillStyle = gradient;
      ctx.fillText(text, centerX, centerY);

      const gloss = ctx.createLinearGradient(0, centerY - fontSize * 0.62, 0, centerY + fontSize * 0.2);
      gloss.addColorStop(0, 'rgba(255,248,196,0.34)');
      gloss.addColorStop(0.65, 'rgba(255,248,196,0)');
      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = gloss;
      ctx.fillText(text, centerX, centerY);
      ctx.restore();

      ctx.lineWidth = Math.max(1.4, fontSize * 0.018);
      ctx.strokeStyle = 'rgba(250,204,21,0.28)';
      ctx.strokeText(text, centerX, centerY);

      boundsRef.current = {
        left: centerX - measured / 2,
        right: centerX + measured / 2 + extrusionDepth,
        width: measured + extrusionDepth,
      };

      const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = image.data;
      const sampleStep = Math.max(1, Math.floor(dpr));
      const nextParticles: Particle[] = [];

      for (let y = 0; y < canvas.height; y += sampleStep) {
        for (let x = 0; x < canvas.width; x += sampleStep) {
          const idx = (y * canvas.width + x) * 4;
          const a = data[idx + 3];
          if (a < 40) continue;
          const baseAlpha = a / 255;
          nextParticles.push({
            x,
            y,
            originX: x,
            originY: y,
            r: data[idx],
            g: data[idx + 1],
            b: data[idx + 2],
            alpha: 0,
            baseAlpha,
            vx: 0,
            vy: 0,
          });
        }
      }

      particlesRef.current = nextParticles;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };

    setup();
    const onResize = () => setup();
    window.addEventListener('resize', onResize);

    const startedAt = performance.now();
    let lastTime = startedAt;

    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      const elapsed = now - startedAt;
      const fadeProgress = Math.max(0, Math.min(1, elapsed / fadeInMs));
      const vaporProgress = Math.max(0, Math.min(1, (elapsed - fadeInMs) / vaporizeMs));
      const { left, width } = boundsRef.current;
      const sweepX = left + width * vaporProgress;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const p of particlesRef.current) {
        if (elapsed < fadeInMs) {
          p.alpha = p.baseAlpha * fadeProgress;
        } else if (p.originX <= sweepX) {
          if (p.vx === 0 && p.vy === 0) {
            const angle = (Math.random() - 0.5) * Math.PI * 0.9;
            const speed = 16 + Math.random() * 40;
            p.vx = Math.cos(angle) * speed;
            p.vy = Math.sin(angle) * speed - (6 + Math.random() * 10);
          }
          p.vx *= 0.985;
          p.vy = p.vy * 0.985 - 2.4 * dt;
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.alpha = Math.max(0, p.alpha - 1.25 * dt);
        } else {
          p.alpha = p.baseAlpha;
        }

        if (p.alpha <= 0.01) continue;
        ctx.fillStyle = `rgba(${p.r},${p.g},${p.b},${p.alpha})`;
        ctx.fillRect(p.x, p.y, Math.max(1.2, dpr), Math.max(1.2, dpr));
      }

      if (!doneRef.current && elapsed >= totalMs + 120) {
        doneRef.current = true;
        onComplete?.();
        return;
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      window.removeEventListener('resize', onResize);
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [fadeInMs, onComplete, text, totalMs, vaporizeMs]);

  return (
    <div
      ref={wrapperRef}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 20,
        background: '#000000',
        overflow: 'hidden',
      }}
    >
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
