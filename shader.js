/* =========================================================
   DoseBot — shader.js
   WebGL soft-particle background for the hero section.
   Falls back to CSS animation on mobile or no-WebGL.
   ========================================================= */
(function () {
  'use strict';

  /* ---- Mobile / feature detection ---- */
  const isMobile = () => window.innerWidth < 768;

  function supportsWebGL() {
    try {
      const c = document.createElement('canvas');
      return !!(c.getContext('webgl') || c.getContext('experimental-webgl'));
    } catch (_) { return false; }
  }

  const canvas = document.getElementById('shaderCanvas');
  if (!canvas) return;

  if (isMobile() || !supportsWebGL()) {
    canvas.style.display = 'none';
    document.body.classList.add('shader-fallback');
    return;
  }

  /* ---- WebGL context ---- */
  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

  /* ---- Vertex shader (full-screen quad) ---- */
  const VS = `
    attribute vec2 a_pos;
    void main() {
      gl_Position = vec4(a_pos, 0.0, 1.0);
    }
  `;

  /* ---- Fragment shader ---- */
  const FS = `
    precision mediump float;

    uniform float  u_time;
    uniform vec2   u_resolution;
    uniform vec2   u_mouse;

    /* ---- Value noise ---- */
    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
    }

    float vnoise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }

    /* ---- Fractal Brownian Motion ---- */
    float fbm(vec2 p) {
      float v = 0.0, amp = 0.5;
      for (int i = 0; i < 5; i++) {
        v   += amp * vnoise(p);
        p   *= 2.0;
        amp *= 0.5;
      }
      return v;
    }

    /* ---- Soft radial particle ---- */
    float particle(vec2 uv, vec2 centre, float radius) {
      float d = length(uv - centre);
      return smoothstep(radius, 0.0, d);
    }

    void main() {
      vec2 uv    = gl_FragCoord.xy / u_resolution;
      vec2 mouse = u_mouse / u_resolution;

      /* Subtle mouse parallax offset */
      vec2 offset = (mouse - 0.5) * 0.03;

      /* Background: #f8fafc in 0-1 space */
      vec3 col = vec3(0.973, 0.980, 0.988);

      /* Soft teal tint from fbm */
      float wave = fbm(uv * 2.5 + u_time * 0.06);
      vec3 tealTint = vec3(0.878, 0.973, 0.945);  /* #e0f7f1 */
      vec3 blueTint = vec3(0.910, 0.957, 0.992);  /* #e8f4fd */
      col = mix(col, tealTint, wave * 0.10);
      col = mix(col, blueTint, fbm(uv * 3.0 - u_time * 0.04) * 0.06);

      /* Floating particles — 10 of them */
      float particles = 0.0;
      for (int i = 0; i < 10; i++) {
        float fi = float(i);
        float speed   = 0.025 + fi * 0.004;
        float xNoise  = vnoise(vec2(fi * 1.73, u_time * 0.08));
        float yNoise  = vnoise(vec2(fi * 2.41, u_time * 0.05 + 10.0));

        vec2 pos = vec2(
          fract(xNoise + fi * 0.1 + u_time * speed * 0.3),
          fract(1.0 - u_time * speed + yNoise * 0.5)
        );

        /* Mouse parallax per-particle */
        pos += offset * (0.5 + fi * 0.07);

        float sz = 0.06 + fi * 0.012;
        float glow = particle(uv, pos, sz);
        particles += glow * (0.18 + fi * 0.01);
      }

      vec3 particleCol = mix(tealTint, blueTint, 0.5);
      col = mix(col, particleCol, clamp(particles, 0.0, 0.3));

      gl_FragColor = vec4(col, 1.0);
    }
  `;

  /* ---- Shader helpers ---- */
  function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.warn('[DoseBot shader]', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function createProgram(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.warn('[DoseBot shader]', gl.getProgramInfoLog(p));
      return null;
    }
    return p;
  }

  const vs  = createShader(gl.VERTEX_SHADER, VS);
  const fs  = createShader(gl.FRAGMENT_SHADER, FS);
  if (!vs || !fs) { canvas.style.display = 'none'; return; }

  const prog = createProgram(vs, fs);
  if (!prog) { canvas.style.display = 'none'; return; }

  /* ---- Full-screen quad ---- */
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  const aPos    = gl.getAttribLocation(prog, 'a_pos');
  const uTime   = gl.getUniformLocation(prog, 'u_time');
  const uRes    = gl.getUniformLocation(prog, 'u_resolution');
  const uMouse  = gl.getUniformLocation(prog, 'u_mouse');

  /* ---- Mouse tracking ---- */
  let mouseX = 0, mouseY = 0;
  window.addEventListener('mousemove', e => {
    mouseX = e.clientX;
    mouseY = window.innerHeight - e.clientY;  // flip Y for WebGL coords
  }, { passive: true });

  /* ---- Resize ---- */
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width  = window.innerWidth  * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width  = window.innerWidth  + 'px';
    canvas.style.height = window.innerHeight + 'px';
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  resize();
  window.addEventListener('resize', resize, { passive: true });

  /* ---- Reduced motion: skip animation ---- */
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reducedMotion) { canvas.style.display = 'none'; return; }

  /* ---- Render loop ---- */
  let startTime = null;
  function render(timestamp) {
    if (!startTime) startTime = timestamp;
    const t = (timestamp - startTime) * 0.001;  // seconds

    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1f(uTime, t);
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform2f(uMouse, mouseX * (window.devicePixelRatio || 1), mouseY * (window.devicePixelRatio || 1));

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
})();
