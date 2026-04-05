import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Activity, ArrowRight, Eye, EyeOff } from 'lucide-react';
import { motion } from 'framer-motion';
import DecryptedText from '../DecryptedText';

export type TravelConnectAuthMode = 'signin' | 'signup';

type TravelConnectSubmitPayload = {
  mode: TravelConnectAuthMode;
  email: string;
  username: string;
  password: string;
};

type TravelConnectSigninProps = {
  mode: TravelConnectAuthMode;
  onModeChange: (mode: TravelConnectAuthMode) => void;
  onSubmit: (payload: TravelConnectSubmitPayload) => void | Promise<void>;
  loading?: boolean;
  error?: string;
  notice?: string;
};

type RoutePoint = {
  x: number;
  y: number;
  delay: number;
};

type Route = {
  start: RoutePoint;
  end: RoutePoint;
  color: string;
};

const ROUTES: Route[] = [
  { start: { x: 0.13, y: 0.27, delay: 0 }, end: { x: 0.34, y: 0.18, delay: 0 }, color: '#00ff9c' },
  { start: { x: 0.34, y: 0.18, delay: 0.6 }, end: { x: 0.48, y: 0.25, delay: 0.6 }, color: '#00cc7d' },
  { start: { x: 0.18, y: 0.68, delay: 1.1 }, end: { x: 0.43, y: 0.37, delay: 1.1 }, color: '#ff4ecd' },
  { start: { x: 0.70, y: 0.24, delay: 0.4 }, end: { x: 0.50, y: 0.62, delay: 0.4 }, color: '#00ff9c' },
  { start: { x: 0.53, y: 0.62, delay: 1.3 }, end: { x: 0.73, y: 0.56, delay: 1.3 }, color: '#ff4ecd' },
];

function generateDots(width: number, height: number) {
  const dots: Array<{ x: number; y: number; radius: number; opacity: number }> = [];
  const gap = 12;

  for (let x = 0; x < width; x += gap) {
    for (let y = 0; y < height; y += gap) {
      const inMapShape =
        ((x < width * 0.26 && x > width * 0.06) && (y < height * 0.41 && y > height * 0.1)) ||
        ((x < width * 0.27 && x > width * 0.15) && (y < height * 0.8 && y > height * 0.43)) ||
        ((x < width * 0.48 && x > width * 0.31) && (y < height * 0.34 && y > height * 0.16)) ||
        ((x < width * 0.54 && x > width * 0.37) && (y < height * 0.68 && y > height * 0.34)) ||
        ((x < width * 0.74 && x > width * 0.45) && (y < height * 0.5 && y > height * 0.12)) ||
        ((x < width * 0.82 && x > width * 0.66) && (y < height * 0.8 && y > height * 0.58));

      if (inMapShape && Math.random() > 0.3) {
        dots.push({
          x,
          y,
          radius: 1,
          opacity: Math.random() * 0.5 + 0.1,
        });
      }
    }
  }

  return dots;
}

function DotMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const syncSize = () => {
      const { width, height } = container.getBoundingClientRect();
      const nextWidth = Math.max(1, Math.floor(width));
      const nextHeight = Math.max(1, Math.floor(height));
      setDimensions({ width: nextWidth, height: nextHeight });
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    };

    syncSize();

    const resizeObserver = new ResizeObserver(() => syncSize());
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    if (!dimensions.width || !dimensions.height) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) return;

    const dots = generateDots(dimensions.width, dimensions.height);
    let animationFrameId = 0;
    let startedAt = performance.now();

    const drawDots = () => {
      context.clearRect(0, 0, dimensions.width, dimensions.height);
      dots.forEach((dot) => {
        context.beginPath();
        context.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        context.fillStyle = `rgba(255, 255, 255, ${dot.opacity})`;
        context.fill();
      });
    };

    const drawRoutes = () => {
      const currentTime = (performance.now() - startedAt) / 1000;

      ROUTES.forEach((route) => {
        const startX = route.start.x * dimensions.width;
        const startY = route.start.y * dimensions.height;
        const endX = route.end.x * dimensions.width;
        const endY = route.end.y * dimensions.height;
        const elapsed = currentTime - route.start.delay;
        if (elapsed <= 0) return;

        const duration = 3;
        const progress = Math.min(elapsed / duration, 1);
        const x = startX + (endX - startX) * progress;
        const y = startY + (endY - startY) * progress;

        context.beginPath();
        context.moveTo(startX, startY);
        context.lineTo(x, y);
        context.strokeStyle = route.color;
        context.lineWidth = 1.5;
        context.stroke();

        context.beginPath();
        context.arc(startX, startY, 3, 0, Math.PI * 2);
        context.fillStyle = route.color;
        context.fill();

        context.beginPath();
        context.arc(x, y, 3, 0, Math.PI * 2);
        context.fillStyle = '#00ff9c';
        context.fill();

        context.beginPath();
        context.arc(x, y, 6, 0, Math.PI * 2);
        context.fillStyle = 'rgba(0, 255, 156, 0.3)';
        context.fill();

        if (progress === 1) {
          context.beginPath();
          context.arc(endX, endY, 3, 0, Math.PI * 2);
          context.fillStyle = route.color;
          context.fill();
        }
      });
    };

    const animate = () => {
      drawDots();
      drawRoutes();

      if ((performance.now() - startedAt) / 1000 > 15) {
        startedAt = performance.now();
      }

      animationFrameId = window.requestAnimationFrame(animate);
    };

    animate();

    return () => window.cancelAnimationFrame(animationFrameId);
  }, [dimensions]);

  return (
    <div ref={containerRef} className="travel-connect-map-stage">
      <canvas ref={canvasRef} className="travel-connect-map-canvas" />
    </div>
  );
}

export default function TravelConnectSignin({
  mode,
  onModeChange,
  onSubmit,
  loading = false,
  error = '',
  notice = '',
}: TravelConnectSigninProps) {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const isSignup = mode === 'signup';

  useEffect(() => {
    setIsPasswordVisible(false);
  }, [mode]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void onSubmit({
      mode,
      email: email.trim(),
      username: username.trim(),
      password,
    });
  };

  return (
    <div className="travel-connect-shell">
      <motion.div
        initial={{ opacity: 0, scale: 0.97, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
        className="travel-connect-card"
      >
        <div className="travel-connect-map">
          <DotMap />
          <div className="travel-connect-map-overlay" aria-hidden="true">
            <div className="travel-connect-logo">
              <h2 style={{ fontSize: '3rem', letterSpacing: '-0.02em', background: 'linear-gradient(180deg, #fff 0%, #00ff9c 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                <DecryptedText
                  text="FinFlux"
                  speed={45}
                  maxIterations={12}
                  characters="01!@#%"
                  animateOn="view"
                  revealDirection="center"
                  sequential
                />
              </h2>
            </div>
            <p style={{ marginTop: '16px' }}>Secure finance intelligence for faster decisions, structured insights, and memory-aware analysis.</p>
          </div>

          <div className="travel-connect-stat travel-connect-stat-top">
            <span>Revenue Pulse</span>
            <strong>+18.4%</strong>
          </div>
          <div className="travel-connect-stat travel-connect-stat-mid">
            <span>Risk Scan</span>
            <strong>LOW</strong>
          </div>
          <div className="travel-connect-stat travel-connect-stat-bottom">
            <span>Insight Pipeline</span>
            <strong>LIVE</strong>
          </div>
        </div>

        <div className="travel-connect-form">
          <div className="travel-connect-brand-row">
            <Activity size={16} /> FinFlux
          </div>

          {/* Toggle hidden by default; use link instead */}

          <h1>{isSignup ? 'Create your account' : 'Welcome back'}</h1>
          <p>{isSignup ? 'Add your details to create a secure workspace.' : 'Sign in to your account.'}</p>

          <div className="travel-connect-subtext">
            {isSignup ? (
              <button type="button" className="travel-connect-link" onClick={() => onModeChange('signin')}>
                Already have an account? Sign in
              </button>
            ) : (
              <div>
                <button type="button" className="travel-connect-link" onClick={() => onModeChange('signup')}>
                  Or, <span style={{ textDecoration: 'underline' }}>Sign up new</span>
                </button>
              </div>
            )}
          </div>

          {notice && <div className="travel-connect-status success">{notice}</div>}
          {error && <div className="travel-connect-status error">{error}</div>}

          <form className="travel-connect-fields" onSubmit={handleSubmit}>
            <label className="travel-connect-field">
              <span>Email</span>
              <input
                className="travel-connect-input"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="Enter your email address"
                autoComplete="email"
                required
              />
            </label>

            {isSignup && (
              <label className="travel-connect-field">
                <span>Username</span>
                <input
                  className="travel-connect-input"
                  type="text"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="Enter your username"
                  autoComplete="username"
                  required
                />
              </label>
            )}

            <label className="travel-connect-field">
              <span>Password</span>
              <div className="travel-connect-password-row">
                <input
                  className="travel-connect-input travel-connect-password-input"
                  type={isPasswordVisible ? 'text' : 'password'}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Enter your password"
                  autoComplete={isSignup ? 'new-password' : 'current-password'}
                  required
                />
                <button
                  type="button"
                  className="travel-connect-eye"
                  onClick={() => setIsPasswordVisible((value) => !value)}
                  aria-label={isPasswordVisible ? 'Hide password' : 'Show password'}
                >
                  {isPasswordVisible ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </label>

            <motion.button
              type="submit"
              className="travel-connect-submit"
              disabled={loading || !email.trim() || !password.trim() || (isSignup && !username.trim())}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
              onHoverStart={() => setIsHovered(true)}
              onHoverEnd={() => setIsHovered(false)}
            >
              <span>
                {loading ? 'Working...' : isSignup ? 'Sign Up' : 'Sign In'}
                <ArrowRight size={16} />
              </span>
              {isHovered && <span className="travel-connect-sheen" />}
            </motion.button>
          </form>
        </div>
      </motion.div>
    </div>
  );
}