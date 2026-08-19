// Input aggregation: keyboard, pointer-locked mouse, wheel, touch and gamepad,
// normalised into a small polled state the camera reads once per frame.

export class Input {
  constructor(canvas) {
    this.canvas = canvas;
    this.keys = new Set();
    this.pressedThisFrame = new Set();
    this.mouseDX = 0;
    this.mouseDY = 0;
    this.wheel = 0;
    this.pointerLocked = false;
    this.buttons = new Set();
    this.touches = new Map();
    this.pinch = 0;
    this.enabled = true;
    this.lastPointer = { x: 0, y: 0, inside: false };
    this.clickHandlers = [];
    this._suppressNextClick = false;

    this._bind();
  }

  _bind() {
    const c = this.canvas;

    addEventListener('keydown', (e) => {
      if (!this.enabled) return;
      if (e.target !== document.body && e.target.tagName === 'INPUT') return;
      const k = e.code;
      if (!this.keys.has(k)) this.pressedThisFrame.add(k);
      this.keys.add(k);
      // Keep the browser from scrolling or triggering quick-find while flying.
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab', 'Slash'].includes(k)) e.preventDefault();
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => { this.keys.clear(); this.buttons.clear(); });

    c.addEventListener('mousedown', (e) => {
      if (!this.enabled) return;
      this.buttons.add(e.button);
      if (e.button === 0 && !this.pointerLocked) {
        this._dragStart = { x: e.clientX, y: e.clientY, t: performance.now() };
      }
    });
    addEventListener('mouseup', (e) => {
      this.buttons.delete(e.button);
      if (e.button === 0 && this._dragStart && !this.pointerLocked) {
        const dx = e.clientX - this._dragStart.x, dy = e.clientY - this._dragStart.y;
        const moved = Math.hypot(dx, dy);
        const dt = performance.now() - this._dragStart.t;
        // A short, still press is a click on the world; a drag is a look.
        if (moved < 5 && dt < 400 && !this._suppressNextClick) {
          for (const h of this.clickHandlers) h(e.clientX, e.clientY);
        }
        this._dragStart = null;
        this._suppressNextClick = false;
      }
    });

    addEventListener('mousemove', (e) => {
      if (!this.enabled) return;
      this.lastPointer.x = e.clientX; this.lastPointer.y = e.clientY;
      if (this.pointerLocked) {
        this.mouseDX += e.movementX || 0;
        this.mouseDY += e.movementY || 0;
      } else if (this.buttons.has(0)) {
        this.mouseDX += e.movementX || 0;
        this.mouseDY += e.movementY || 0;
      }
    });

    c.addEventListener('wheel', (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      this.wheel += Math.sign(e.deltaY) * Math.min(3, Math.abs(e.deltaY) / 50 + 0.5);
    }, { passive: false });

    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === c;
    });

    c.addEventListener('contextmenu', (e) => e.preventDefault());

    /* ------------------------------------------------------------ touch -- */
    c.addEventListener('touchstart', (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      for (const t of e.changedTouches) this.touches.set(t.identifier, { x: t.clientX, y: t.clientY, x0: t.clientX, y0: t.clientY });
      if (this.touches.size === 2) this._pinch0 = this._touchDistance();
    }, { passive: false });

    c.addEventListener('touchmove', (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      let dx = 0, dy = 0, n = 0;
      for (const t of e.changedTouches) {
        const p = this.touches.get(t.identifier);
        if (!p) continue;
        dx += t.clientX - p.x; dy += t.clientY - p.y; n++;
        p.x = t.clientX; p.y = t.clientY;
      }
      if (this.touches.size === 1 && n) { this.mouseDX += dx; this.mouseDY += dy; }
      if (this.touches.size === 2) {
        const d = this._touchDistance();
        this.pinch += (d - (this._pinch0 || d)) * 0.01;
        this._pinch0 = d;
        this.twoFingerDY = dy / Math.max(1, n);
      }
    }, { passive: false });

    const endTouch = (e) => {
      for (const t of e.changedTouches) this.touches.delete(t.identifier);
    };
    c.addEventListener('touchend', endTouch);
    c.addEventListener('touchcancel', endTouch);
  }

  _touchDistance() {
    const t = [...this.touches.values()];
    if (t.length < 2) return 0;
    return Math.hypot(t[0].x - t[1].x, t[0].y - t[1].y);
  }

  onWorldClick(fn) { this.clickHandlers.push(fn); }
  suppressClick() { this._suppressNextClick = true; }

  requestPointerLock() {
    if (!this.pointerLocked) this.canvas.requestPointerLock?.();
  }
  exitPointerLock() {
    if (this.pointerLocked) document.exitPointerLock?.();
  }

  down(code) { return this.keys.has(code); }
  pressed(code) { return this.pressedThisFrame.has(code); }
  axis(negCode, posCode) { return (this.down(posCode) ? 1 : 0) - (this.down(negCode) ? 1 : 0); }

  // Called once at the end of each frame to consume per-frame deltas.
  endFrame() {
    this.mouseDX = 0; this.mouseDY = 0; this.wheel = 0; this.pinch = 0;
    this.twoFingerDY = 0;
    this.pressedThisFrame.clear();
  }

  get isTouch() { return this.touches.size > 0; }
}
