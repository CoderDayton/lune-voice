import { useEffect, useRef } from 'react';

type Rgb = [number, number, number];

const VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `
precision mediump float;

varying vec2 v_uv;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_opacity;
uniform float u_intensity;
uniform vec3 u_primary;
uniform vec3 u_secondary;
uniform vec3 u_accent;

float softBlob(vec2 p, vec2 center, float radius) {
  float d = distance(p, center);
  return smoothstep(radius, 0.0, d);
}

void main() {
  vec2 uv = v_uv;
  vec2 p = uv - 0.5;
  p.x *= u_resolution.x / max(u_resolution.y, 1.0);

  float t = u_time * 0.035;
  float waveA = sin((p.x * 2.1 + p.y * 1.3) * 3.14159 + t * 1.7) * 0.5 + 0.5;
  float waveB = sin((p.x * -1.4 + p.y * 2.2) * 3.14159 - t * 1.2) * 0.5 + 0.5;

  vec2 c1 = vec2(-0.52 + sin(t * 0.7) * 0.05, -0.22 + cos(t * 0.5) * 0.04);
  vec2 c2 = vec2(0.46 + cos(t * 0.6) * 0.05, 0.28 + sin(t * 0.8) * 0.04);
  float blob = softBlob(p, c1, 0.78) * 0.55 + softBlob(p, c2, 0.68) * 0.45;

  vec3 color = mix(u_primary, u_accent, waveA * 0.45);
  color = mix(color, u_secondary, waveB * 0.22);

  float vignette = smoothstep(0.95, 0.12, length(p));
  float alpha = (blob * 0.55 + waveA * waveB * 0.12) * vignette;
  alpha = clamp(alpha * u_opacity * u_intensity, 0.0, u_opacity);

  gl_FragColor = vec4(color, alpha);
}
`;

interface ShaderProgram {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  uniforms: {
    resolution: WebGLUniformLocation | null;
    time: WebGLUniformLocation | null;
    opacity: WebGLUniformLocation | null;
    intensity: WebGLUniformLocation | null;
    primary: WebGLUniformLocation | null;
    secondary: WebGLUniformLocation | null;
    accent: WebGLUniformLocation | null;
  };
  buffer: WebGLBuffer;
}

interface NavigatorWithHints extends Navigator {
  deviceMemory?: number;
  connection?: {
    saveData?: boolean;
    effectiveType?: string;
  };
}

export function ShaderBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || shouldDisableShader()) return;

    const targetCanvas = canvas;
    const shader = createShaderProgram(targetCanvas);
    if (!shader) return;

    const { gl, uniforms } = shader;
    let frame = 0;
    let lastFrame = 0;
    let running = true;
    let width = 0;
    let height = 0;
    let opacity = 0.08;
    let intensity = 0.55;
    let primary: Rgb = [0.55, 0.46, 0.86];
    let secondary: Rgb = [0.48, 0.66, 0.72];
    let accent: Rgb = [0.48, 0.74, 0.72];

    function readTheme() {
      const styles = getComputedStyle(document.documentElement);
      const scheme = document.documentElement.getAttribute('data-color-scheme');
      primary = softenColor(readCssColor(styles, '--color-primary', primary), scheme);
      secondary = softenColor(readCssColor(styles, '--color-secondary', secondary), scheme);
      accent = softenColor(readCssColor(styles, '--color-accent', accent), scheme);
      opacity = scheme === 'light' ? 0.038 : 0.065;
      intensity = scheme === 'light' ? 0.38 : 0.48;
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const nextWidth = Math.max(1, Math.floor(window.innerWidth * dpr));
      const nextHeight = Math.max(1, Math.floor(window.innerHeight * dpr));
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      targetCanvas.width = width;
      targetCanvas.height = height;
      gl.viewport(0, 0, width, height);
    }

    function render(now: number) {
      if (!running) return;
      frame = requestAnimationFrame(render);
      if (document.hidden || now - lastFrame < 33) return;
      lastFrame = now;

      resize();
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.resolution, width, height);
      gl.uniform1f(uniforms.time, now * 0.001);
      gl.uniform1f(uniforms.opacity, opacity);
      gl.uniform1f(uniforms.intensity, intensity);
      gl.uniform3fv(uniforms.primary, primary);
      gl.uniform3fv(uniforms.secondary, secondary);
      gl.uniform3fv(uniforms.accent, accent);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    readTheme();
    resize();
    frame = requestAnimationFrame(render);

    const themeObserver = new MutationObserver(readTheme);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-color-scheme'],
    });

    window.addEventListener('resize', resize);

    return () => {
      running = false;
      cancelAnimationFrame(frame);
      themeObserver.disconnect();
      window.removeEventListener('resize', resize);
      gl.deleteProgram(shader.program);
      gl.deleteBuffer(shader.buffer);
    };
  }, []);

  return <canvas ref={canvasRef} className="app-shader-background" />;
}

function shouldDisableShader() {
  if (typeof window === 'undefined') return true;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return true;

  const nav = navigator as NavigatorWithHints;
  if (nav.connection?.saveData) return true;
  if (nav.deviceMemory !== undefined && nav.deviceMemory <= 4) return true;
  if (navigator.hardwareConcurrency !== undefined && navigator.hardwareConcurrency <= 4) {
    return true;
  }
  return false;
}

function createShaderProgram(canvas: HTMLCanvasElement): ShaderProgram | null {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: 'low-power',
    premultipliedAlpha: true,
  });
  if (!gl) return null;

  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  if (!vertex || !fragment) {
    if (vertex) gl.deleteShader(vertex);
    if (fragment) gl.deleteShader(fragment);
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    return null;
  }
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }

  const buffer = gl.createBuffer();
  if (!buffer) {
    gl.deleteProgram(program);
    return null;
  }

  const activateProgram = gl.useProgram.bind(gl);
  activateProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const position = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.clearColor(0, 0, 0, 0);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);

  return {
    gl,
    program,
    buffer,
    uniforms: {
      resolution: gl.getUniformLocation(program, 'u_resolution'),
      time: gl.getUniformLocation(program, 'u_time'),
      opacity: gl.getUniformLocation(program, 'u_opacity'),
      intensity: gl.getUniformLocation(program, 'u_intensity'),
      primary: gl.getUniformLocation(program, 'u_primary'),
      secondary: gl.getUniformLocation(program, 'u_secondary'),
      accent: gl.getUniformLocation(program, 'u_accent'),
    },
  };
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function readCssColor(styles: CSSStyleDeclaration, token: string, fallback: Rgb): Rgb {
  const value = styles.getPropertyValue(token).trim();
  return parseShaderCssColor(value) ?? fallback;
}

function softenColor(color: Rgb, scheme: string | null): Rgb {
  const luma = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;
  const neutral = scheme === 'light' ? Math.min(1, luma * 1.08 + 0.04) : luma * 0.78;
  const saturation = scheme === 'light' ? 0.28 : 0.34;
  return [
    neutral + (color[0] - neutral) * saturation,
    neutral + (color[1] - neutral) * saturation,
    neutral + (color[2] - neutral) * saturation,
  ];
}

export function parseShaderCssColor(value: string): Rgb | null {
  if (value.startsWith('rgb')) return parseRgb(value);
  if (value.startsWith('oklch')) return parseOklch(value);
  return null;
}

function parseRgb(value: string): Rgb | null {
  const channels = value
    .match(/[\d.]+/g)
    ?.slice(0, 3)
    .map(Number);
  if (!channels || channels.length < 3 || channels.some((n) => !Number.isFinite(n))) return null;
  return [channels[0] / 255, channels[1] / 255, channels[2] / 255];
}

function parseOklch(value: string): Rgb | null {
  const match = value.match(/oklch\(\s*([\d.]+%?)\s+([\d.]+)\s+([\d.]+)/);
  if (!match) return null;

  const l = match[1].endsWith('%') ? Number.parseFloat(match[1]) / 100 : Number(match[1]);
  const c = Number(match[2]);
  const h = (Number(match[3]) * Math.PI) / 180;
  if (![l, c, h].every(Number.isFinite)) return null;

  return oklabToSrgb(l, Math.cos(h) * c, Math.sin(h) * c);
}

function oklabToSrgb(l: number, a: number, b: number): Rgb {
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = l - 0.0894841775 * a - 1.291485548 * b;

  const l3 = l_ * l_ * l_;
  const m3 = m_ * m_ * m_;
  const s3 = s_ * s_ * s_;

  const r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
  const g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
  const bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3;

  return [linearToSrgb(r), linearToSrgb(g), linearToSrgb(bl)];
}

function linearToSrgb(channel: number) {
  const clamped = Math.min(1, Math.max(0, channel));
  if (clamped <= 0.0031308) return clamped * 12.92;
  return 1.055 * clamped ** (1 / 2.4) - 0.055;
}
