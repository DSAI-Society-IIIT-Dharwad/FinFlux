import { useEffect, useMemo, useRef } from 'react';
import type { CSSProperties } from 'react';
import {
  Mesh,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';

import './FloatingLines.css';

type WaveName = 'top' | 'middle' | 'bottom';

type FloatingLinesProps = {
  enabledWaves?: WaveName[];
  lineCount?: number | number[];
  lineDistance?: number | number[];
  bendRadius?: number;
  bendStrength?: number;
  interactive?: boolean;
  parallax?: boolean;
  parallaxStrength?: number;
  animationSpeed?: number;
  mixBlendMode?: CSSProperties['mixBlendMode'];
  linesGradient?: string[];
  topWavePosition?: { x?: number; y?: number; rotate?: number };
  middleWavePosition?: { x?: number; y?: number; rotate?: number };
  bottomWavePosition?: { x?: number; y?: number; rotate?: number };
};

const MAX_GRADIENT_STOPS = 8;

const vertexShader = `
precision highp float;

varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
precision highp float;

uniform float iTime;
uniform vec3  iResolution;
uniform float animationSpeed;

uniform bool enableTop;
uniform bool enableMiddle;
uniform bool enableBottom;

uniform int topLineCount;
uniform int middleLineCount;
uniform int bottomLineCount;

uniform float topLineDistance;
uniform float middleLineDistance;
uniform float bottomLineDistance;

uniform vec3 topWavePosition;
uniform vec3 middleWavePosition;
uniform vec3 bottomWavePosition;

uniform vec2 iMouse;
uniform bool interactive;
uniform float bendRadius;
uniform float bendStrength;
uniform float bendInfluence;

uniform bool parallax;
uniform vec2 parallaxOffset;

uniform vec3 lineGradient[8];
uniform int lineGradientCount;

const vec3 BLACK = vec3(0.0);
const vec3 GREEN_A = vec3(0.30, 0.73, 0.09);
const vec3 GREEN_B = vec3(0.10, 0.33, 0.06);
const vec3 GREEN_C = vec3(0.52, 0.87, 0.30);

mat2 rotate(float r) {
  return mat2(cos(r), sin(r), -sin(r), cos(r));
}

vec3 background_color(vec2 uv) {
  vec3 col = vec3(0.0);
  float y = sin(uv.x - 0.2) * 0.25 - 0.12;
  float m = uv.y - y;
  col += mix(BLACK, GREEN_B, smoothstep(0.0, 1.2, abs(m)));
  col += mix(GREEN_A, BLACK, smoothstep(0.0, 1.0, abs(m - 0.8)));
  return col * 0.6;
}

vec3 getLineColor(float t, vec3 baseColor) {
  if (lineGradientCount <= 0) {
    return baseColor;
  }
  vec3 gradientColor;
  if (lineGradientCount == 1) {
    gradientColor = lineGradient[0];
  } else {
    float clampedT = clamp(t, 0.0, 0.9999);
    float scaled = clampedT * float(lineGradientCount - 1);
    int idx = int(floor(scaled));
    float f = fract(scaled);
    int idx2 = min(idx + 1, lineGradientCount - 1);
    vec3 c1 = lineGradient[idx];
    vec3 c2 = lineGradient[idx2];
    gradientColor = mix(c1, c2, f);
  }
  return gradientColor * 0.8;
}

float wave(vec2 uv, float offset, vec2 screenUv, vec2 mouseUv, bool shouldBend) {
  float time = iTime * animationSpeed;
  float x_offset = offset;
  float x_movement = time * 0.1;
  float amp = sin(offset + time * 0.2) * 0.26;
  float y = sin(uv.x + x_offset + x_movement) * amp;

  if (shouldBend) {
    vec2 d = screenUv - mouseUv;
    float influence = exp(-dot(d, d) * bendRadius);
    float bendOffset = (mouseUv.y - screenUv.y) * influence * bendStrength * bendInfluence;
    y += bendOffset;
  }

  float m = uv.y - y;
  return 0.02 / max(abs(m) + 0.01, 1e-3) + 0.01;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 baseUv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
  baseUv.y *= -1.0;

  if (parallax) {
    baseUv += parallaxOffset;
  }

  vec3 col = vec3(0.0);
  vec3 background = lineGradientCount > 0 ? vec3(0.0) : background_color(baseUv);
  vec2 mouseUv = vec2(0.0);
  if (interactive) {
    mouseUv = (2.0 * iMouse - iResolution.xy) / iResolution.y;
    mouseUv.y *= -1.0;
  }

  if (enableBottom) {
    for (int i = 0; i < bottomLineCount; ++i) {
      float fi = float(i);
      float t = fi / max(float(bottomLineCount - 1), 1.0);
      vec3 lineCol = getLineColor(t, background);
      float angle = bottomWavePosition.z * log(length(baseUv) + 1.0);
      vec2 ruv = baseUv * rotate(angle);
      col += lineCol * wave(ruv + vec2(bottomLineDistance * fi + bottomWavePosition.x, bottomWavePosition.y), 1.4 + 0.2 * fi, baseUv, mouseUv, interactive) * 0.2;
    }
  }

  if (enableMiddle) {
    for (int i = 0; i < middleLineCount; ++i) {
      float fi = float(i);
      float t = fi / max(float(middleLineCount - 1), 1.0);
      vec3 lineCol = getLineColor(t, background);
      float angle = middleWavePosition.z * log(length(baseUv) + 1.0);
      vec2 ruv = baseUv * rotate(angle);
      col += lineCol * wave(ruv + vec2(middleLineDistance * fi + middleWavePosition.x, middleWavePosition.y), 2.0 + 0.15 * fi, baseUv, mouseUv, interactive);
    }
  }

  if (enableTop) {
    for (int i = 0; i < topLineCount; ++i) {
      float fi = float(i);
      float t = fi / max(float(topLineCount - 1), 1.0);
      vec3 lineCol = getLineColor(t, background);
      float angle = topWavePosition.z * log(length(baseUv) + 1.0);
      vec2 ruv = baseUv * rotate(angle);
      ruv.x *= -1.0;
      col += lineCol * wave(ruv + vec2(topLineDistance * fi + topWavePosition.x, topWavePosition.y), 1.0 + 0.2 * fi, baseUv, mouseUv, interactive) * 0.1;
    }
  }

  fragColor = vec4(col, 1.0);
}

void main() {
  vec4 color = vec4(0.0);
  mainImage(color, gl_FragCoord.xy);
  gl_FragColor = color;
}
`;

function hexToVec3(hex: string): Vector3 {
  let value = hex.trim();
  if (value.startsWith('#')) value = value.slice(1);

  let r = 255;
  let g = 255;
  let b = 255;

  if (value.length === 3) {
    r = parseInt(value[0] + value[0], 16);
    g = parseInt(value[1] + value[1], 16);
    b = parseInt(value[2] + value[2], 16);
  } else if (value.length === 6) {
    r = parseInt(value.slice(0, 2), 16);
    g = parseInt(value.slice(2, 4), 16);
    b = parseInt(value.slice(4, 6), 16);
  }

  return new Vector3(r / 255, g / 255, b / 255);
}

export default function FloatingLines({
  linesGradient,
  enabledWaves = ['top', 'middle', 'bottom'],
  lineCount = [6],
  lineDistance = [5],
  topWavePosition,
  middleWavePosition,
  bottomWavePosition = { x: 2.0, y: -0.7, rotate: -1 },
  animationSpeed = 1,
  interactive = true,
  bendRadius = 5.0,
  bendStrength = -0.5,
  parallax = true,
  parallaxStrength = 0.18,
  mixBlendMode = 'screen',
}: FloatingLinesProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const targetMouseRef = useRef(new Vector2(-1000, -1000));
  const currentMouseRef = useRef(new Vector2(-1000, -1000));
  const targetInfluenceRef = useRef(0);
  const currentInfluenceRef = useRef(0);
  const targetParallaxRef = useRef(new Vector2(0, 0));
  const currentParallaxRef = useRef(new Vector2(0, 0));

  const getLineCount = (waveType: WaveName) => {
    if (typeof lineCount === 'number') return lineCount;
    if (!enabledWaves.includes(waveType)) return 0;
    const index = enabledWaves.indexOf(waveType);
    return lineCount[index] ?? 6;
  };

  const getLineDistance = (waveType: WaveName) => {
    if (typeof lineDistance === 'number') return lineDistance;
    if (!enabledWaves.includes(waveType)) return 0.1;
    const index = enabledWaves.indexOf(waveType);
    return lineDistance[index] ?? 0.1;
  };

  const topLineCount = enabledWaves.includes('top') ? getLineCount('top') : 0;
  const middleLineCount = enabledWaves.includes('middle') ? getLineCount('middle') : 0;
  const bottomLineCount = enabledWaves.includes('bottom') ? getLineCount('bottom') : 0;

  const topLineDistance = enabledWaves.includes('top') ? getLineDistance('top') * 0.01 : 0.01;
  const middleLineDistance = enabledWaves.includes('middle') ? getLineDistance('middle') * 0.01 : 0.01;
  const bottomLineDistance = enabledWaves.includes('bottom') ? getLineDistance('bottom') * 0.01 : 0.01;

  const gradientStops = useMemo(() => {
    const fallback = ['#071325', '#1a4410', '#4cbb17', '#9be564'];
    const source = (linesGradient?.length ? linesGradient : fallback).slice(0, MAX_GRADIENT_STOPS);
    return source;
  }, [linesGradient]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let active = true;

    const scene = new Scene();
    const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    camera.position.z = 1;

    const renderer = new WebGLRenderer({ antialias: false, alpha: false, powerPreference: 'low-power' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    const uniforms = {
      iTime: { value: 0 },
      iResolution: { value: new Vector3(1, 1, 1) },
      animationSpeed: { value: animationSpeed },

      enableTop: { value: enabledWaves.includes('top') },
      enableMiddle: { value: enabledWaves.includes('middle') },
      enableBottom: { value: enabledWaves.includes('bottom') },

      topLineCount: { value: topLineCount },
      middleLineCount: { value: middleLineCount },
      bottomLineCount: { value: bottomLineCount },

      topLineDistance: { value: topLineDistance },
      middleLineDistance: { value: middleLineDistance },
      bottomLineDistance: { value: bottomLineDistance },

      topWavePosition: { value: new Vector3(topWavePosition?.x ?? 10.0, topWavePosition?.y ?? 0.5, topWavePosition?.rotate ?? -0.4) },
      middleWavePosition: { value: new Vector3(middleWavePosition?.x ?? 5.0, middleWavePosition?.y ?? 0.0, middleWavePosition?.rotate ?? 0.2) },
      bottomWavePosition: { value: new Vector3(bottomWavePosition?.x ?? 2.0, bottomWavePosition?.y ?? -0.7, bottomWavePosition?.rotate ?? 0.4) },

      iMouse: { value: new Vector2(-1000, -1000) },
      interactive: { value: interactive },
      bendRadius: { value: bendRadius },
      bendStrength: { value: bendStrength },
      bendInfluence: { value: 0 },

      parallax: { value: parallax },
      parallaxOffset: { value: new Vector2(0, 0) },

      lineGradient: { value: Array.from({ length: MAX_GRADIENT_STOPS }, () => new Vector3(1, 1, 1)) },
      lineGradientCount: { value: 0 },
    };

    uniforms.lineGradientCount.value = gradientStops.length;
    gradientStops.forEach((hex, i) => {
      const color = hexToVec3(hex);
      uniforms.lineGradient.value[i].set(color.x, color.y, color.z);
    });

    const material = new ShaderMaterial({
      uniforms,
      vertexShader,
      fragmentShader,
    });

    const geometry = new PlaneGeometry(2, 2);
    const mesh = new Mesh(geometry, material);
    scene.add(mesh);

    const setSize = () => {
      if (!active) return;
      const width = container.clientWidth || 1;
      const height = container.clientHeight || 1;
      renderer.setSize(width, height, false);
      const canvasWidth = renderer.domElement.width;
      const canvasHeight = renderer.domElement.height;
      uniforms.iResolution.value.set(canvasWidth, canvasHeight, 1);
    };

    setSize();

    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { if (active) setSize(); }) : null;
    if (ro) ro.observe(container);

    const handlePointerMove = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const dpr = renderer.getPixelRatio();

      targetMouseRef.current.set(x * dpr, (rect.height - y) * dpr);
      targetInfluenceRef.current = 1.0;

      if (parallax) {
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const offsetX = (x - centerX) / rect.width;
        const offsetY = -(y - centerY) / rect.height;
        targetParallaxRef.current.set(offsetX * parallaxStrength, offsetY * parallaxStrength);
      }
    };

    const handlePointerLeave = () => { targetInfluenceRef.current = 0.0; };

    if (interactive) {
      renderer.domElement.addEventListener('pointermove', handlePointerMove);
      renderer.domElement.addEventListener('pointerleave', handlePointerLeave);
    }

    let raf = 0;
    const startedAt = performance.now();
    let lastTime = startedAt;
    let lastRenderAt = startedAt;
    const targetFrameMs = 1000 / 30;

    const renderLoop = () => {
      if (!active) return;

      const now = performance.now();
      if (now - lastRenderAt < targetFrameMs) {
        raf = requestAnimationFrame(renderLoop);
        return;
      }
      lastRenderAt = now;
      const deltaSeconds = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      uniforms.iTime.value = (now - startedAt) / 1000;

      if (interactive) {
        currentMouseRef.current.lerp(targetMouseRef.current, 0.05);
        uniforms.iMouse.value.copy(currentMouseRef.current);
        currentInfluenceRef.current += (targetInfluenceRef.current - currentInfluenceRef.current) * 0.05;
        uniforms.bendInfluence.value = currentInfluenceRef.current;
      }

      if (parallax) {
        currentParallaxRef.current.lerp(targetParallaxRef.current, 0.05);
        uniforms.parallaxOffset.value.copy(currentParallaxRef.current);
      }

      void deltaSeconds;

      renderer.render(scene, camera);
      raf = requestAnimationFrame(renderLoop);
    };

    renderLoop();

    return () => {
      active = false;
      cancelAnimationFrame(raf);
      if (ro) ro.disconnect();
      if (interactive) {
        renderer.domElement.removeEventListener('pointermove', handlePointerMove);
        renderer.domElement.removeEventListener('pointerleave', handlePointerLeave);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentElement) {
        renderer.domElement.parentElement.removeChild(renderer.domElement);
      }
    };
  }, [animationSpeed, bendRadius, bendStrength, enabledWaves, gradientStops, interactive, lineDistance, lineCount, middleWavePosition, parallax, parallaxStrength, bottomWavePosition, topWavePosition, topLineDistance, topLineCount, middleLineDistance, middleLineCount, bottomLineDistance, bottomLineCount]);

  return (
    <div
      ref={containerRef}
      className="floating-lines-container"
      style={{ mixBlendMode }}
    />
  );
}
