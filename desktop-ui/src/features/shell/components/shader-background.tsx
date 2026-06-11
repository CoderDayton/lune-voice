import { useEffect, useRef } from 'react';

type Rgb = [number, number, number];
type ColorScheme = 'dark' | 'light';

// Fallbacks only — live values come from the theme tokens (--color-primary …)
// graded through gradeThemeColor. The opacity/intensity/grain numbers here are
// what actually drive the look, since the app defines no --shader-* overrides.
const DEFAULT_DARK: ShaderPalette = {
  primary: [0.6, 0.49, 0.88],
  secondary: [0.45, 0.41, 0.6],
  accent: [0.42, 0.76, 0.8],
  opacity: 0.55,
  intensity: 1.0,
  grain: 0.022,
};

const DEFAULT_LIGHT: ShaderPalette = {
  primary: [0.42, 0.3, 0.72],
  secondary: [0.34, 0.3, 0.52],
  accent: [0.2, 0.52, 0.58],
  opacity: 0.16,
  intensity: 1.0,
  grain: 0.01,
};

const VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_uv;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_opacity;
uniform float u_intensity;
uniform float u_grain;
uniform vec3 u_primary;
uniform vec3 u_secondary;
uniform vec3 u_accent;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);

  float a = hash12(i + vec2(0.0, 0.0));
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Rotation between octaves breaks up axis-aligned artefacts in the noise.
const mat2 M2 = mat2(0.8, 0.6, -0.6, 0.8);

float fbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.5;
  for (int i = 0; i < 5; i++) {
    value += amplitude * noise(p);
    p = M2 * p * 2.0 + 13.1;
    amplitude *= 0.5;
  }
  return value;
}

// Soft radial blob with a tunable edge falloff — anchors colour into corners.
float softOrb(vec2 p, vec2 center, float radius, float falloff) {
  float d = length((p - center) / radius);
  return pow(clamp(1.0 - d, 0.0, 1.0), falloff);
}

void main() {
  float aspect = u_resolution.x / max(u_resolution.y, 1.0);
  vec2 uv = v_uv;
  vec2 p = uv - 0.5;
  p.x *= aspect;

  float t = u_time * 0.03;

  // Iñigo-Quílez-style domain warp: fbm fed through fbm twice for an organic,
  // slowly evolving flow that never visibly tiles or repeats.
  vec2 q = vec2(
    fbm(p * 1.15 + vec2(0.0, t * 0.55)),
    fbm(p * 1.15 + vec2(4.6, -t * 0.45))
  );
  vec2 r = vec2(
    fbm(p * 1.35 + 1.6 * q + vec2(7.3, 2.8) + t * 0.35),
    fbm(p * 1.35 + 1.6 * q + vec2(1.7, 9.2) - t * 0.28)
  );
  float f = fbm(p * 1.55 + 2.0 * r);

  // Weave the three theme tones through the warped field.
  vec3 color = mix(u_secondary, u_primary, clamp(f * 1.4, 0.0, 1.0));
  color = mix(color, u_accent, clamp(dot(r, r) * 0.85, 0.0, 1.0));
  color = mix(color, u_primary, clamp((q.x - 0.3) * 0.9, 0.0, 1.0) * 0.5);

  // Drifting luminous pools pull colour into the corners so the aurora frames
  // the content rather than washing flatly across it.
  vec2 driftA = vec2(sin(t * 0.70), cos(t * 0.50)) * 0.08;
  vec2 driftB = vec2(cos(t * 0.45), sin(t * 0.62)) * 0.07;
  vec2 driftC = vec2(sin(t * 0.33), cos(t * 0.58)) * 0.06;
  float pools =
      softOrb(p, vec2(-0.78, 0.42) + driftA, 1.15, 1.5)
    + softOrb(p, vec2(0.85, -0.52) + driftB, 1.25, 1.5)
    + softOrb(p, vec2(0.55, 0.66) + driftC, 0.95, 1.7);

  float density = pow(clamp(f, 0.0, 1.0), 1.25);
  float shape = density * (0.45 + 0.95 * pools);

  // Keep the field alive across the whole frame, easing off only at the far
  // corners so it dissolves smoothly into the body gradient.
  float vignette = smoothstep(1.35, 0.20, length(p));
  shape *= mix(0.6, 1.0, vignette);

  float alpha = clamp(shape * u_intensity, 0.0, 1.0) * u_opacity;

  // Dither both colour and alpha — essential to avoid visible banding on the
  // dark, low-contrast gradient this composites over.
  float grain = (hash12(gl_FragCoord.xy + fract(u_time) * 157.0) - 0.5) * u_grain;
  color = clamp(color + grain, vec3(0.0), vec3(1.0));
  alpha = clamp(alpha + grain * 0.6, 0.0, 1.0);

  gl_FragColor = vec4(color, alpha);
}
`;

const FRAME_INTERVAL = 1000 / 30;
// Fixed phase used to render a single pleasant still frame under reduced-motion.
const STATIC_PHASE = 8.0;

interface ShaderPalette {
  primary: Rgb;
  secondary: Rgb;
  accent: Rgb;
  opacity: number;
  intensity: number;
  grain: number;
}

interface ShaderProgram {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  uniforms: {
    resolution: WebGLUniformLocation | null;
    time: WebGLUniformLocation | null;
    opacity: WebGLUniformLocation | null;
    intensity: WebGLUniformLocation | null;
    grain: WebGLUniformLocation | null;
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
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let frame = 0;
    let lastFrame = 0;
    let running = true;
    let width = 0;
    let height = 0;
    let palette = readThemePalette();

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
      const viewportWidth = window.visualViewport?.width ?? window.innerWidth;
      const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
      const nextWidth = Math.max(1, Math.floor(viewportWidth * dpr));
      const nextHeight = Math.max(1, Math.floor(viewportHeight * dpr));

      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      targetCanvas.width = width;
      targetCanvas.height = height;
      gl.viewport(0, 0, width, height);
    }

    function draw(timeSeconds: number) {
      resize();
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.resolution, width, height);
      gl.uniform1f(uniforms.time, timeSeconds);
      gl.uniform1f(uniforms.opacity, palette.opacity);
      gl.uniform1f(uniforms.intensity, palette.intensity);
      gl.uniform1f(uniforms.grain, palette.grain);
      gl.uniform3fv(uniforms.primary, palette.primary);
      gl.uniform3fv(uniforms.secondary, palette.secondary);
      gl.uniform3fv(uniforms.accent, palette.accent);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function loop(now: number) {
      if (!running) return;
      frame = requestAnimationFrame(loop);
      if (document.hidden || now - lastFrame < FRAME_INTERVAL) return;
      lastFrame = now;
      draw(now * 0.001);
    }

    function readTheme() {
      palette = readThemePalette();
      // The animated loop picks up the new palette on its next frame; the
      // static path has to redraw explicitly.
      if (reducedMotion && running) draw(STATIC_PHASE);
    }

    function handleResize() {
      if (reducedMotion && running) draw(STATIC_PHASE);
      else resize();
    }

    resize();
    if (reducedMotion) draw(STATIC_PHASE);
    else frame = requestAnimationFrame(loop);

    const themeObserver = new MutationObserver(readTheme);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'style', 'data-theme', 'data-color-scheme'],
    });

    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener?.('change', readTheme);
    window.addEventListener('resize', handleResize);
    window.visualViewport?.addEventListener('resize', handleResize);

    return () => {
      running = false;
      cancelAnimationFrame(frame);
      themeObserver.disconnect();
      media.removeEventListener?.('change', readTheme);
      window.removeEventListener('resize', handleResize);
      window.visualViewport?.removeEventListener('resize', handleResize);
      gl.deleteProgram(shader.program);
      gl.deleteBuffer(shader.buffer);
    };
  }, []);

  return <canvas ref={canvasRef} className="app-shader-background" />;
}

function shouldDisableShader() {
  if (typeof window === 'undefined') return true;

  const nav = navigator as NavigatorWithHints;
  if (nav.connection?.saveData) return true;
  if (nav.deviceMemory !== undefined && nav.deviceMemory <= 4) return true;
  if (navigator.hardwareConcurrency !== undefined && navigator.hardwareConcurrency <= 4)
    return true;

  return false;
}

function createShaderProgram(canvas: HTMLCanvasElement): ShaderProgram | null {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: 'low-power',
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
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

  // Aliased so Biome's react-hooks lint doesn't mistake gl.useProgram for a hook.
  const activateProgram = gl.useProgram.bind(gl);
  activateProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const position = gl.getAttribLocation(program, 'a_position');
  if (position < 0) {
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    return null;
  }

  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.clearColor(0, 0, 0, 0);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.disable(gl.BLEND);

  return {
    gl,
    program,
    buffer,
    uniforms: {
      resolution: gl.getUniformLocation(program, 'u_resolution'),
      time: gl.getUniformLocation(program, 'u_time'),
      opacity: gl.getUniformLocation(program, 'u_opacity'),
      intensity: gl.getUniformLocation(program, 'u_intensity'),
      grain: gl.getUniformLocation(program, 'u_grain'),
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

function readThemePalette(): ShaderPalette {
  const root = document.documentElement;
  const styles = getComputedStyle(root);
  const scheme = inferColorScheme(root, styles);
  const defaults = scheme === 'light' ? DEFAULT_LIGHT : DEFAULT_DARK;

  return {
    primary: gradeThemeColor(readCssColor(styles, '--color-primary', defaults.primary), scheme),
    secondary: gradeThemeColor(
      readCssColor(styles, '--color-secondary', defaults.secondary),
      scheme,
    ),
    accent: gradeThemeColor(readCssColor(styles, '--color-accent', defaults.accent), scheme),
    opacity: readCssNumber(
      styles,
      '--shader-opacity',
      defaults.opacity,
      0,
      scheme === 'light' ? 0.34 : 0.85,
    ),
    intensity: readCssNumber(styles, '--shader-intensity', defaults.intensity, 0, 1.25),
    grain: readCssNumber(styles, '--shader-grain', defaults.grain, 0, 0.05),
  };
}

function inferColorScheme(root: HTMLElement, styles: CSSStyleDeclaration): ColorScheme {
  const explicit =
    `${root.getAttribute('data-color-scheme') ?? ''} ${root.getAttribute('data-theme') ?? ''} ${root.className}`.toLowerCase();
  if (explicit.includes('light')) return 'light';
  if (explicit.includes('dark')) return 'dark';

  const cssScheme = styles.getPropertyValue('color-scheme').toLowerCase();
  if (cssScheme.includes('light') && !cssScheme.includes('dark')) return 'light';
  if (cssScheme.includes('dark')) return 'dark';

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function readCssColor(styles: CSSStyleDeclaration, token: string, fallback: Rgb): Rgb {
  const value = styles.getPropertyValue(token).trim();
  return parseShaderCssColor(value) ?? fallback;
}

function readCssNumber(
  styles: CSSStyleDeclaration,
  token: string,
  fallback: number,
  min: number,
  max: number,
) {
  const parsed = Number.parseFloat(styles.getPropertyValue(token));
  if (!Number.isFinite(parsed)) return fallback;
  return clamp(parsed, min, max);
}

// Tune a raw theme colour for use as an aurora light. Subtlety is owned by the
// alpha shaping in the shader, so here we keep the hue vivid (just bounded) on
// dark, and deepen it for contrast on light.
function gradeThemeColor(color: Rgb, scheme: ColorScheme): Rgb {
  const safe = color.map((channel) => clamp(channel, 0, 1)) as Rgb;

  if (scheme === 'light') {
    const deep = mixRgb(safe, [0.12, 0.11, 0.16], 0.3);
    return saturateRgb(deep, 1.15);
  }

  const vivid = saturateRgb(safe, 1.16);
  return clampLuminance(vivid, 0.74);
}

function saturateRgb(color: Rgb, amount: number): Rgb {
  const luma = luminance(color);
  return [
    clamp(luma + (color[0] - luma) * amount, 0, 1),
    clamp(luma + (color[1] - luma) * amount, 0, 1),
    clamp(luma + (color[2] - luma) * amount, 0, 1),
  ];
}

function clampLuminance(color: Rgb, maxLuma: number): Rgb {
  const luma = luminance(color);
  if (luma <= maxLuma || luma <= 0) return color;
  const scale = maxLuma / luma;
  return [color[0] * scale, color[1] * scale, color[2] * scale];
}

function luminance(color: Rgb) {
  return color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;
}

function mixRgb(a: Rgb, b: Rgb, amount: number): Rgb {
  const t = clamp(amount, 0, 1);
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function parseShaderCssColor(value: string): Rgb | null {
  const color = value.trim().toLowerCase();
  if (!color || color === 'transparent') return null;
  if (color.startsWith('#')) return parseHex(color);
  if (color.startsWith('rgb')) return parseRgb(color);
  if (color.startsWith('hsl')) return parseHsl(color);
  if (color.startsWith('oklch')) return parseOklch(color);
  return null;
}

function parseHex(value: string): Rgb | null {
  const hex = value.slice(1);
  if (![3, 4, 6, 8].includes(hex.length) || /[^0-9a-f]/i.test(hex)) return null;

  const normalized =
    hex.length <= 4
      ? hex
          .slice(0, 3)
          .split('')
          .map((char) => char + char)
          .join('')
      : hex.slice(0, 6);
  const int = Number.parseInt(normalized, 16);
  if (!Number.isFinite(int)) return null;

  return [((int >> 16) & 255) / 255, ((int >> 8) & 255) / 255, (int & 255) / 255];
}

function parseRgb(value: string): Rgb | null {
  const channels = value.match(/-?[\d.]+%?/g)?.slice(0, 3);
  if (!channels || channels.length < 3) return null;

  const parsed = channels.map((channel) => {
    if (channel.endsWith('%')) return clamp(Number.parseFloat(channel) / 100, 0, 1);
    return clamp(Number.parseFloat(channel) / 255, 0, 1);
  });

  if (parsed.some((n) => !Number.isFinite(n))) return null;
  return parsed as Rgb;
}

function parseHsl(value: string): Rgb | null {
  const channels = value.match(/-?[\d.]+%?/g)?.slice(0, 3);
  if (!channels || channels.length < 3) return null;

  const h = normalizeHue(Number.parseFloat(channels[0]));
  const s = parsePercent(channels[1]);
  const l = parsePercent(channels[2]);
  if (![h, s, l].every(Number.isFinite)) return null;

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const m = l - c / 2;

  let rgb: Rgb;
  if (hp < 1) rgb = [c, x, 0];
  else if (hp < 2) rgb = [x, c, 0];
  else if (hp < 3) rgb = [0, c, x];
  else if (hp < 4) rgb = [0, x, c];
  else if (hp < 5) rgb = [x, 0, c];
  else rgb = [c, 0, x];

  return [rgb[0] + m, rgb[1] + m, rgb[2] + m];
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

function parsePercent(value: string) {
  if (value.endsWith('%')) return clamp(Number.parseFloat(value) / 100, 0, 1);
  return clamp(Number.parseFloat(value), 0, 1);
}

function normalizeHue(value: number) {
  return ((value % 360) + 360) % 360;
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
  const clamped = clamp(channel, 0, 1);
  if (clamped <= 0.0031308) return clamped * 12.92;
  return 1.055 * clamped ** (1 / 2.4) - 0.055;
}
