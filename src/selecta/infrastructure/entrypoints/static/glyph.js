// Selecta's glyph: a round matrix of dots that shows it listening, thinking and speaking.
// Each dot eases toward the state's brightness, so state changes blend instead of jumping.
export class Glyph {
  constructor(canvas, { dots = 25, base = 0.14, live = "#d71921" } = {}) {
    this.canvas = canvas;
    this.base = base;
    this.live = live;
    this.state = "idle";
    this.input = 0;
    this.smooth = 0;
    this.points = [];
    const half = (dots - 1) / 2;
    for (let row = 0; row < dots; row++) {
      for (let col = 0; col < dots; col++) {
        const x = (col - half) / half, y = (row - half) / half, d = Math.hypot(x, y);
        if (d <= 1.02) this.points.push({ x, y, d, a: Math.atan2(y, x), b: 0 });
      }
    }
    this.step = 1 / half;
    this.running = false;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.canvas.clientWidth * ratio);
    this.canvas.height = Math.round(this.canvas.clientHeight * ratio);
    this.run();
  }

  set(state) {
    this.state = state;
    this.run();
  }

  level(value) {
    this.input = Math.max(0, Math.min(1, value));
  }

  run() {
    if (this.running) return;
    this.running = true;
    requestAnimationFrame(t => this.frame(t));
  }

  target(p, t) {
    const level = this.smooth;
    switch (this.state) {
      case "listening": {
        const reach = 0.12 + level * 0.95;
        return Math.max(0, Math.min(1, (reach - p.d) / 0.18)) * (0.55 + 0.45 * level);
      }
      case "thinking": {
        if (p.d < 0.62) return 0;
        const head = (t / 1100) * Math.PI * 2;
        const behind = (((head - p.a) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        return behind < 1.4 ? (1 - behind / 1.4) ** 1.6 : 0;
      }
      case "speaking": {
        const wave = 0.5 + 0.5 * Math.cos(p.d * 9 - t / 140);
        return Math.min(1, (0.08 + level * 1.1) * wave * (1.15 - p.d * 0.5));
      }
      default:
        return 0;
    }
  }

  frame(t) {
    this.smooth += (this.input - this.smooth) * 0.3;
    const { width: w, height: h } = this.canvas;
    const ctx = this.canvas.getContext("2d");
    const radius = Math.min(w, h) / 2, dot = radius * this.step * 0.36;
    ctx.clearRect(0, 0, w, h);
    let moving = this.state !== "idle";
    for (const p of this.points) {
      const goal = this.target(p, t);
      p.b += (goal - p.b) * (goal > p.b ? 0.45 : 0.16);
      if (Math.abs(goal - p.b) > 0.004) moving = true;
      const lit = p.b;
      ctx.globalAlpha = this.base + (1 - this.base) * lit;
      ctx.fillStyle = this.state === "listening" && p.d < this.step * 0.5 ? this.live : "#fff";
      ctx.beginPath();
      ctx.arc(w / 2 + p.x * (radius - dot), h / 2 + p.y * (radius - dot), dot * (0.6 + 0.4 * lit), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    this.running = moving;
    if (moving) requestAnimationFrame(next => this.frame(next));
  }
}
