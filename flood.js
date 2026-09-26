// Sublevel-set flood of a heading, with its live persistence diagram.
//
// The heading's letters are carved into a height field
//   f = 1 − 0.62·(G₁ ∗ L) − 0.30·(G₂ ∗ L) + η − D
// (L = letter mask, G = Gaussian blurs, η = random field whose Fourier
// coefficients follow Ornstein–Uhlenbeck processes, D = what the pointer digs).
// Water fills the sublevel set {f ≤ t}. The inset plots the persistence
// diagram of that filtration on the pixel grid (cubical V-construction, as in
// the flood paper): H0 by union-find over 4-neighbours, H1 by duality from
// H0 of −f over 8-neighbours with an "outside" node on the border.
(() => {
  'use strict';

  const reduceMQ = matchMedia('(prefers-reduced-motion: reduce)');
  const INTRO_RISE = 2.5, INTRO_FALL = 1.2;           // seconds
  const MODES = 16, ETA_STD = 0.045, ETA_THETA = 0.25; // idle terrain noise
  const T_STD = 0.014, T_THETA = 0.35;                 // idle water level
  const DIG_DEPTH = 0.5, DIG_SIGMA = 9, DIG_TAU = 2.4;  // pointer (CSS px, s)

  function rng(seed) { // mulberry32
    return () => {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

  // ---------- persistence (checked against a brute-force boundary reduction) ----------
  function makePH(N) {
    const order = new Uint32Array(N), rank = new Uint32Array(N);
    const parent = new Int32Array(N + 1), birth = new Float64Array(N + 1), added = new Uint8Array(N);
    // one native numeric sort of packed keys: value quantized to 2^-22, then index
    let M = 1; while (M < N) M *= 2;
    const keys = new Float64Array(N);
    function find(x) {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    }
    function sort(f) {
      let lo = Infinity;
      for (let i = 0; i < N; i++) if (f[i] < lo) lo = f[i];
      for (let i = 0; i < N; i++) keys[i] = Math.round((f[i] - lo) * 4194304) * M + i;
      keys.sort();
      for (let i = 0; i < N; i++) order[i] = keys[i] % M;
    }
    function ranks(desc) {
      for (let i = 0; i < N; i++) rank[order[desc ? N - 1 - i : i]] = i;
    }
    // merge the roots of p and q; the younger class dies at value v
    function merge(p, q, v, out, flip) {
      let a = find(p), b = find(q);
      if (a === b) return;
      if (birth[a] < birth[b] || (birth[a] === birth[b] && a !== N && (b === N || rank[a] < rank[b]))) { const s = a; a = b; b = s; }
      if (v > birth[a]) { if (flip) out.push(-v, -birth[a]); else out.push(birth[a], v); }
      parent[a] = b;
    }
    return function compute(f, W, H, h0, h1) {
      h0.length = 0; h1.length = 0;
      sort(f); ranks(false); added.fill(0);
      for (let k = 0; k < N; k++) {
        const p = order[k], x = p % W, v = f[p];
        parent[p] = p; birth[p] = v; added[p] = 1;
        if (x > 0 && added[p - 1]) merge(p, p - 1, v, h0, false);
        if (x < W - 1 && added[p + 1]) merge(p, p + 1, v, h0, false);
        if (p >= W && added[p - W]) merge(p, p - W, v, h0, false);
        if (p + W < N && added[p + W]) merge(p, p + W, v, h0, false);
      }
      const min = f[order[0]];
      ranks(true); added.fill(0);
      parent[N] = N; birth[N] = -Infinity;
      for (let k = N - 1; k >= 0; k--) {
        const p = order[k], x = p % W, y = (p / W) | 0, g = -f[p];
        parent[p] = p; birth[p] = g; added[p] = 1;
        if (x === 0 || y === 0 || x === W - 1 || y === H - 1) merge(p, N, g, h1, true);
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if ((!dx && !dy) || nx < 0 || nx >= W) continue;
            const q = ny * W + nx;
            if (added[q]) merge(p, q, g, h1, true);
          }
        }
      }
      return min;
    };
  }

  function blur(src, W, H, sigma) {
    const r = Math.max(1, Math.ceil(sigma * 3)), k = new Float32Array(2 * r + 1);
    let s = 0;
    for (let i = -r; i <= r; i++) s += k[i + r] = Math.exp(-i * i / (2 * sigma * sigma));
    for (let i = 0; i < k.length; i++) k[i] /= s;
    const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) { const xx = x + i; if (xx >= 0 && xx < W) acc += k[i + r] * src[y * W + xx]; }
      tmp[y * W + x] = acc;
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) { const yy = y + i; if (yy >= 0 && yy < H) acc += k[i + r] * tmp[yy * W + x]; }
      out[y * W + x] = acc;
    }
    let m = 0;
    for (let i = 0; i < out.length; i++) if (out[i] > m) m = out[i];
    if (m > 0) for (let i = 0; i < out.length; i++) out[i] /= m;
    return out;
  }

  const probe = document.createElement('canvas').getContext('2d');
  function rgb(css, fallback) {
    probe.fillStyle = fallback; probe.fillStyle = css.trim() || fallback;
    const c = probe.fillStyle;
    if (c[0] === '#') return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
    return c.match(/[\d.]+/g).slice(0, 3).map(Number);
  }

  class Flood {
    constructor(root) {
      this.root = root;
      this.field = root.querySelector('.flood-field');
      this.text = root.querySelector('[data-flood-text]');
      this.pd = root.querySelector('.flood-pd canvas');
      this.readout = root.querySelector('[data-flood-readout]');
      this.fctx = this.field.getContext('2d');
      this.pctx = this.pd && this.pd.getContext('2d');
      this.h0 = []; this.h1 = [];
      this.intro = !reduceMQ.matches;
      this.visible = true; this.onscreen = true;
      this.pointer = null; this.lastPointer = 0;
      this.frame = 0; this.raf = 0;
      this.readColors();
      this.build();
      this.introStart = performance.now();
      this.last = this.introStart;

      new ResizeObserver(() => { clearTimeout(this.rt); this.rt = setTimeout(() => this.rebuild(), 120); }).observe(root);
      (document.fonts ? Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 1500))]) : Promise.resolve())
        .then(() => { this.build(); this.draw(); });
      new MutationObserver(() => { this.readColors(); this.draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { this.readColors(); this.draw(); });
      reduceMQ.addEventListener('change', () => { this.intro = false; this.tLevel = this.tRest; this.u = 0; this.draw(); this.settle(); this.schedule(); });
      document.addEventListener('visibilitychange', () => { this.visible = !document.hidden; this.schedule(); });
      new IntersectionObserver(e => { this.onscreen = e[0].isIntersecting; this.schedule(); }).observe(this.field);

      const skip = () => { if (this.intro) { this.intro = false; this.tLevel = this.tRest; } this.settle(); };
      if (!this.intro) this.settle();
      for (const ev of ['keydown', 'wheel', 'touchstart', 'pointerdown', 'scroll']) addEventListener(ev, skip, { once: true, passive: true });

      const move = e => {
        if (reduceMQ.matches) return;
        if (e.pointerType !== 'mouse' && e.type === 'pointermove' && !e.pressure && !this.touching) return;
        const r = this.field.getBoundingClientRect();
        const p = { x: e.clientX - r.left, y: e.clientY - r.top };
        this.dig(this.pointer || p, p);
        this.pointer = p; this.lastPointer = performance.now();
        this.schedule();
      };
      root.addEventListener('pointerdown', e => { this.touching = true; this.pointer = null; move(e); });
      root.addEventListener('pointermove', move);
      const up = () => { this.touching = false; this.pointer = null; };
      root.addEventListener('pointerup', up);
      root.addEventListener('pointercancel', up);
      root.addEventListener('pointerleave', up);

      this.schedule();
    }

    // tells the page the flood has peaked, so text can surface as it drains
    settle() {
      if (this.settled) return;
      this.settled = true;
      dispatchEvent(new Event('flood:settle'));
    }

    readColors() {
      const cs = getComputedStyle(this.root);
      this.c = {
        water: rgb(cs.getPropertyValue('--water'), '#2f6694'),
        shore: cs.getPropertyValue('--shore').trim() || '#244c70',
        contour: cs.getPropertyValue('--contour').trim() || '#cfd5cc',
        h0: cs.getPropertyValue('--h0').trim() || '#244c70',
        h1: cs.getPropertyValue('--h1').trim() || '#b3303c',
        paper: cs.getPropertyValue('--paper').trim() || '#faf9f6',
        line: cs.getPropertyValue('--line').trim() || '#d9ddd7',
        muted: cs.getPropertyValue('--muted').trim() || '#656b66',
      };
    }

    rebuild() {
      const r = this.field.getBoundingClientRect();
      if (Math.abs(r.width - this.cssW) < 1 && Math.abs(r.height - this.cssH) < 1) return;
      this.build();
      this.draw();
    }

    build() {
      const r = this.field.getBoundingClientRect();
      const W0 = this.cssW = r.width, H0 = this.cssH = r.height;
      const dpr = this.dpr = Math.min(2, devicePixelRatio || 1);
      this.field.width = Math.round(W0 * dpr); this.field.height = Math.round(H0 * dpr);
      // about 11k grid cells whatever the width, so small screens keep the letter strokes resolved
      const cell = this.cell = Math.max(3, Math.min(5, Math.sqrt(W0 * H0 / 11000)));
      const W = this.W = Math.ceil(W0 / cell), H = this.H = Math.ceil(H0 / cell), N = this.N = W * H;

      // letter mask at grid resolution, from the heading's own glyph positions
      const oc = document.createElement('canvas'); oc.width = W; oc.height = H;
      const o = oc.getContext('2d');
      const cs = getComputedStyle(this.text);
      o.scale(1 / cell, 1 / cell);
      o.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      o.textBaseline = 'alphabetic'; o.fillStyle = '#000';
      const m = o.measureText('Mg');
      const asc = m.fontBoundingBoxAscent || parseFloat(cs.fontSize) * 0.9;
      const walker = document.createTreeWalker(this.text, NodeFilter.SHOW_TEXT);
      const range = document.createRange();
      for (let node; (node = walker.nextNode());) {
        const s = node.textContent;
        for (let i = 0; i < s.length; i++) {
          if (!s[i].trim()) continue;
          range.setStart(node, i); range.setEnd(node, i + 1);
          const b = range.getClientRects()[0];
          if (b) o.fillText(s[i], b.left - r.left, b.top - r.top + asc);
        }
      }
      const px = o.getImageData(0, 0, W, H).data;
      const L = new Float32Array(N);
      let letterCells = 0;
      for (let i = 0; i < N; i++) { L[i] = px[4 * i + 3] / 255; if (L[i] > 0.5) letterCells++; }
      const fs = parseFloat(cs.fontSize);
      const b1 = blur(L, W, H, 0.035 * fs / cell), b2 = blur(L, W, H, 0.2 * fs / cell);
      this.base = new Float32Array(N);
      for (let i = 0; i < N; i++) this.base[i] = 1 - 0.62 * b1[i] - 0.30 * b2[i];
      const sorted = Float32Array.from(this.base).sort();
      this.tRest = sorted[Math.min(N - 1, Math.floor(N * Math.min(0.5, 1.35 * letterCells / N)))];
      this.lo = sorted[0] - 0.04;
      this.hi = 1 + 3.2 * ETA_STD;
      if (this.tLevel === undefined || !this.intro) this.tLevel = this.tRest;

      // OU random field: η(x) = Σ a_k cos(k·x) + b_k sin(k·x), separable in x and y
      const R = rng(1599);
      this.kx = []; this.cx = []; this.sx = []; this.cy = []; this.sy = [];
      this.a = new Float64Array(MODES); this.b = new Float64Array(MODES);
      const sd = ETA_STD / Math.sqrt(MODES);
      for (let k = 0; k < MODES; k++) {
        const lambda = 22 + 70 * R(), ang = Math.PI * R(), w = 2 * Math.PI / lambda;
        const kx = w * Math.cos(ang), ky = w * Math.sin(ang);
        const cx = new Float32Array(W), sx = new Float32Array(W), cy = new Float32Array(H), sy = new Float32Array(H);
        for (let x = 0; x < W; x++) { cx[x] = Math.cos(kx * (x + 0.5) * cell); sx[x] = Math.sin(kx * (x + 0.5) * cell); }
        for (let y = 0; y < H; y++) { cy[y] = Math.cos(ky * (y + 0.5) * cell); sy[y] = Math.sin(ky * (y + 0.5) * cell); }
        this.cx.push(cx); this.sx.push(sx); this.cy.push(cy); this.sy.push(sy);
        this.a[k] = sd * gauss(R); this.b[k] = sd * gauss(R);
      }
      this.sd = sd; this.R = R; this.u = 0;
      this.D = new Float32Array(N);
      this.f = new Float32Array(N);
      this.ph = makePH(N);
      this.img = new ImageData(W, H);
      this.wc = document.createElement('canvas'); this.wc.width = W; this.wc.height = H;

      // soft edges so the map fades into the page
      const fade = this.fade = document.createElement('canvas');
      fade.width = this.field.width; fade.height = this.field.height;
      const g = fade.getContext('2d'), fx = 56 * dpr, fy = 30 * dpr, FW = fade.width, FH = fade.height;
      let gr = g.createLinearGradient(0, 0, FW, 0);
      gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(Math.min(.45, fx / FW), '#000');
      gr.addColorStop(Math.max(.55, 1 - fx / FW), '#000'); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(0, 0, FW, FH);
      g.globalCompositeOperation = 'destination-in';
      gr = g.createLinearGradient(0, 0, 0, FH);
      gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(Math.min(.45, fy / FH), '#000');
      gr.addColorStop(Math.max(.55, 1 - fy / FH), '#000'); gr.addColorStop(1, 'rgba(0,0,0,0)');
      g.fillStyle = gr; g.fillRect(0, 0, FW, FH);

      if (this.pd) {
        const s = this.pd.getBoundingClientRect().width;
        this.pdS = s; this.pd.width = Math.round(s * dpr); this.pd.height = Math.round(s * dpr);
      }
      this.updateField();
    }

    dig(p0, p1) {
      const { W, H, cell, D } = this;
      const sig = DIG_SIGMA / cell, rad = Math.ceil(3 * sig);
      const len = Math.hypot(p1.x - p0.x, p1.y - p0.y), steps = Math.max(1, Math.ceil(len / (DIG_SIGMA * 0.5)));
      for (let s = 1; s <= steps; s++) {
        const cx = (p0.x + (p1.x - p0.x) * s / steps) / cell - 0.5, cy = (p0.y + (p1.y - p0.y) * s / steps) / cell - 0.5;
        const x0 = Math.max(0, Math.floor(cx - rad)), x1 = Math.min(W - 1, Math.ceil(cx + rad));
        const y0 = Math.max(0, Math.floor(cy - rad)), y1 = Math.min(H - 1, Math.ceil(cy + rad));
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          const d2 = (x - cx) ** 2 + (y - cy) ** 2, v = DIG_DEPTH * Math.exp(-d2 / (2 * sig * sig));
          const i = y * W + x;
          if (v > D[i]) D[i] = v;
        }
      }
    }

    // exact OU step for every Fourier coefficient and for the water level
    stepOU(dt) {
      const e = Math.exp(-ETA_THETA * dt), s = this.sd * Math.sqrt(1 - e * e), R = this.R;
      for (let k = 0; k < MODES; k++) { this.a[k] = this.a[k] * e + s * gauss(R); this.b[k] = this.b[k] * e + s * gauss(R); }
      const et = Math.exp(-T_THETA * dt);
      this.u = this.u * et + T_STD * Math.sqrt(1 - et * et) * gauss(R);
      const ed = Math.exp(-dt / DIG_TAU), D = this.D;
      for (let i = 0; i < D.length; i++) D[i] *= ed;
    }

    updateField() {
      const { W, H, base, D, f, a, b } = this;
      const P = new Float64Array(MODES), Q = new Float64Array(MODES);
      for (let y = 0; y < H; y++) {
        for (let k = 0; k < MODES; k++) {
          const c = this.cy[k][y], s = this.sy[k][y];
          P[k] = a[k] * c + b[k] * s; Q[k] = b[k] * c - a[k] * s;
        }
        for (let x = 0; x < W; x++) {
          let eta = 0;
          for (let k = 0; k < MODES; k++) eta += this.cx[k][x] * P[k] + this.sx[k][x] * Q[k];
          const i = y * W + x;
          f[i] = base[i] + eta - D[i];
        }
      }
      this.min = this.ph(f, W, H, this.h0, this.h1);
    }

    schedule() {
      const run = this.visible && this.onscreen && !reduceMQ.matches;
      if (run && !this.raf) { this.last = performance.now(); this.raf = requestAnimationFrame(t => this.tick(t)); }
      if (!run && reduceMQ.matches) this.draw();
    }

    tick(now) {
      this.raf = 0;
      if (!(this.visible && this.onscreen) || reduceMQ.matches) return;
      const active = this.intro || now - this.lastPointer < 2500;
      this.frame++;
      if (active || this.frame % 2 === 0) { // idle runs at half rate
        // rAF timestamps can precede the performance.now() taken when scheduling
        const dt = Math.max(0, Math.min(0.1, (now - this.last) / 1000)); this.last = now;
        this.stepOU(dt);
        if (this.intro) {
          const s = (now - this.introStart) / 1000, top = this.hi, lo = this.lo;
          if (s < INTRO_RISE) { const p = s / INTRO_RISE; this.tLevel = lo + (top - lo) * (1 - Math.pow(1 - p, 2.2)); }
          else if (s < INTRO_RISE + INTRO_FALL) { const p = (s - INTRO_RISE) / INTRO_FALL; this.tLevel = top + (this.tRest - top) * p * p * (3 - 2 * p); }
          else { this.intro = false; this.tLevel = this.tRest; this.settle(); }
          this.risen = s >= INTRO_RISE;
          if (this.risen) this.settle();
        } else this.tLevel = this.tRest + this.u;
        this.updateField();
        this.draw();
      }
      this.raf = requestAnimationFrame(t => this.tick(t));
    }

    contour(ctx, level) {
      const { W, H, f, cell } = this;
      const X = x => (x + 0.5) * cell, Y = y => (y + 0.5) * cell;
      ctx.beginPath();
      for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
        const i = y * W + x;
        const v0 = f[i], v1 = f[i + 1], v2 = f[i + W + 1], v3 = f[i + W];
        const c = (v0 > level) | (v1 > level) << 1 | (v2 > level) << 2 | (v3 > level) << 3;
        if (c === 0 || c === 15) continue;
        const e = [ // crossing points on edges: top, right, bottom, left
          [X(x + (level - v0) / (v1 - v0)), Y(y)],
          [X(x + 1), Y(y + (level - v1) / (v2 - v1))],
          [X(x + (level - v3) / (v2 - v3)), Y(y + 1)],
          [X(x), Y(y + (level - v0) / (v3 - v0))],
        ];
        const seg = (p, q) => { ctx.moveTo(e[p][0], e[p][1]); ctx.lineTo(e[q][0], e[q][1]); };
        switch (c) {
          case 1: case 14: seg(3, 0); break;
          case 2: case 13: seg(0, 1); break;
          case 3: case 12: seg(3, 1); break;
          case 4: case 11: seg(1, 2); break;
          case 6: case 9: seg(0, 2); break;
          case 7: case 8: seg(3, 2); break;
          // saddles: wet (≤ level) corners that only touch diagonally stay apart,
          // matching the 4-connectivity of the H0 computation
          case 5: seg(0, 1); seg(3, 2); break;
          case 10: seg(3, 0); seg(1, 2); break;
        }
      }
      ctx.stroke();
    }

    draw() {
      if (!this.f) return;
      const ctx = this.fctx, { W, H, f, cell, dpr, img } = this, t = this.tLevel;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.clearRect(0, 0, this.field.width, this.field.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const [r, g, b] = this.c.water, d = img.data;
      for (let i = 0; i < W * H; i++) {
        const depth = t - f[i];
        const al = depth <= 0 ? 0 : Math.min(1, depth / 0.05) * (0.14 + 0.24 * Math.min(1, depth / 0.45));
        d[4 * i] = r; d[4 * i + 1] = g; d[4 * i + 2] = b; d[4 * i + 3] = al * 255;
      }
      this.wc.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.wc, 0, 0, W * cell, H * cell);

      ctx.lineWidth = 1; ctx.strokeStyle = this.c.contour;
      for (const lv of [0.3, 0.5, 0.7, 0.88, 0.94]) this.contour(ctx, lv);
      ctx.lineWidth = 1.3; ctx.strokeStyle = this.c.shore;
      this.contour(ctx, t);

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = 'destination-in';
      ctx.drawImage(this.fade, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      this.drawDiagram(t);
    }

    drawDiagram(t) {
      if (!this.pctx) return;
      const ctx = this.pctx, S = this.pdS, dpr = this.dpr, c = this.c;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, S, S);
      const pad = 4, top = pad + 7, lo = this.lo, hi = this.hi, span = S - pad - top;
      const X = v => pad + (Math.min(hi, v) - lo) / (hi - lo) * span;
      const Y = v => S - pad - (Math.min(hi, v) - lo) / (hi - lo) * span;
      // quadrant of classes alive at level t: birth ≤ t < death
      ctx.fillStyle = `rgba(${c.water.join(',')},0.13)`;
      ctx.fillRect(pad, 0, X(t) - pad, Y(t));
      ctx.strokeStyle = c.muted; ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad, S - pad); ctx.lineTo(S - top + pad, top); ctx.stroke(); // diagonal
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(pad, 3.5); ctx.lineTo(S - pad, 3.5); ctx.stroke(); // ∞ row
      ctx.beginPath(); ctx.moveTo(X(t), Y(t)); ctx.lineTo(X(t), 0); ctx.moveTo(pad, Y(t)); ctx.lineTo(X(t), Y(t)); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;

      const live = this.intro && !this.risen;
      let b0 = this.min <= t ? 1 : 0, b1 = 0;
      const plot = (arr, filled, col) => {
        ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.3;
        for (let i = 0; i < arr.length; i += 2) {
          const bi = arr[i], di = arr[i + 1];
          if (bi <= t && t < di) filled ? b0++ : b1++;
          if (live && bi > t) continue;
          const dd = live ? Math.min(di, t) : di;
          const rad = 1.3 + 2.2 * Math.min(1, (di - bi) / 0.3);
          ctx.beginPath(); ctx.arc(X(bi), Y(dd), rad, 0, 2 * Math.PI);
          filled ? ctx.fill() : ctx.stroke();
        }
      };
      plot(this.h1, false, c.h1);
      plot(this.h0, true, c.h0);
      ctx.fillStyle = c.h0; ctx.beginPath(); ctx.arc(X(this.min), 3.5, 3.4, 0, 2 * Math.PI); ctx.fill(); // essential class

      if (this.readout) {
        const txt = `β<sub>0</sub> = ${b0}&ensp;β<sub>1</sub> = ${b1}`; // b0, b1 are integers
        if (txt !== this.lastTxt) { this.readout.innerHTML = txt; this.lastTxt = txt; }
      }
    }
  }

  const start = () => document.querySelectorAll('[data-flood]').forEach(el => new Flood(el));
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
