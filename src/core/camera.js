// A six-degree-of-freedom flight camera with critically-damped smoothing and a
// logarithmic speed control, so the same rig works from megaparsec to metre.

import {
  v3, v3set, v3copy, v3add, v3sub, v3scale, v3addScaled, v3norm, v3len, v3cross,
  v3transformQuat, v3lerp, v3dot, quat, qmul, qsetAxisAngle, qnorm, qcopy, qslerp,
  qfromTo, m4, m4view, m4perspectiveReverseInfinite, m4mul, m4invert,
  extractFrustum, clamp, damp, mix, DEG,
} from './math.js';

export class Camera {
  constructor(opts = {}) {
    this.position = v3(0, 0, 0);
    this.orientation = quat();

    // Targets the smoothed state chases; input writes here.
    this.targetPosition = v3(0, 0, 0);
    this.targetOrientation = quat();

    this.velocity = v3(0, 0, 0);
    this.fov = (opts.fov || 62) * DEG;
    this.targetFov = this.fov;
    this.near = opts.near || 1e-4;
    this.aspect = 1;

    // Speed is stored as a log so a wheel notch is a constant multiplier.
    this.logSpeed = Math.log(opts.speed || 0.05);
    this.minLogSpeed = Math.log(1e-9);
    this.maxLogSpeed = Math.log(1e4);
    this.boost = 1;

    this.view = m4();
    this.proj = m4();
    this.viewProj = m4();
    this.invViewProj = m4();
    this.frustum = [];

    this.forward = v3(0, 0, -1);
    this.right = v3(1, 0, 0);
    this.up = v3(0, 1, 0);

    this.smoothing = opts.smoothing === undefined ? 18 : opts.smoothing;
    this.lookSensitivity = 0.0022;
    this.rollRate = 1.6;

    // Orbit mode: the camera is bound to a moving anchor.
    this.mode = 'free';
    this.orbit = { center: v3(), distance: 1, minDistance: 0.01, maxDistance: 1e6, yaw: 0, pitch: 0.3 };

    this._tmpQ = quat();
    this._tmpQ2 = quat();
    this._tmpV = v3();
    this._axisX = v3(1, 0, 0);
    this._axisY = v3(0, 1, 0);
    this._axisZ = v3(0, 0, 1);
    this._shake = 0;
    this._shakeSeed = 0;
  }

  get speed() { return Math.exp(this.logSpeed); }
  set speed(v) { this.logSpeed = clamp(Math.log(Math.max(1e-12, v)), this.minLogSpeed, this.maxLogSpeed); }

  setPose(pos, q) {
    v3copy(this.position, pos); v3copy(this.targetPosition, pos);
    qcopy(this.orientation, q); qcopy(this.targetOrientation, q);
  }

  lookAt(target, upHint = null) {
    const dir = v3norm(v3(), v3sub(v3(), target, this.targetPosition));
    this.faceDirection(dir, upHint);
  }

  faceDirection(dir, upHint = null) {
    // Build an orthonormal basis then convert to a quaternion via the
    // shortest-arc from -Z, followed by a roll correction toward `upHint`.
    const f = v3norm(v3(), dir);
    const q = qfromTo(quat(), v3(0, 0, -1), f);
    if (upHint) {
      const currentUp = v3transformQuat(v3(), v3(0, 1, 0), q);
      // Project both ups into the plane perpendicular to f.
      const projUp = v3addScaled(v3(), upHint, f, -v3dot(upHint, f));
      if (v3len(projUp) > 1e-5) {
        v3norm(projUp, projUp);
        const projCur = v3addScaled(v3(), currentUp, f, -v3dot(currentUp, f));
        v3norm(projCur, projCur);
        const roll = qfromTo(quat(), projCur, projUp);
        qmul(q, roll, q);
      }
    }
    qcopy(this.targetOrientation, qnorm(q, q));
  }

  // Applies mouse-look and keyboard translation for one frame.
  fly(input, dt, opts = {}) {
    const speedScale = opts.speedScale || 1;

    if (input.wheel) {
      this.logSpeed = clamp(this.logSpeed - input.wheel * 0.18, this.minLogSpeed, this.maxLogSpeed);
    }
    if (input.pinch) {
      this.logSpeed = clamp(this.logSpeed + input.pinch * 0.5, this.minLogSpeed, this.maxLogSpeed);
    }

    const looking = input.pointerLocked || input.buttons.has(0) || input.isTouch;
    if (looking && (input.mouseDX || input.mouseDY)) {
      const yaw = -input.mouseDX * this.lookSensitivity;
      const pitch = -input.mouseDY * this.lookSensitivity;
      // Rotate in the camera's local frame: no gimbal lock, full roll freedom.
      const q = this.targetOrientation;
      qmul(q, q, qsetAxisAngle(this._tmpQ, this._axisY, yaw));
      qmul(q, q, qsetAxisAngle(this._tmpQ, this._axisX, pitch));
      qnorm(q, q);
    }

    const roll = input.axis('KeyE', 'KeyQ');
    if (roll) {
      qmul(this.targetOrientation, this.targetOrientation,
        qsetAxisAngle(this._tmpQ, this._axisZ, roll * this.rollRate * dt));
      qnorm(this.targetOrientation, this.targetOrientation);
    }

    const fwd = v3transformQuat(v3(), v3(0, 0, -1), this.targetOrientation);
    const rgt = v3transformQuat(v3(), v3(1, 0, 0), this.targetOrientation);
    const upv = v3transformQuat(v3(), v3(0, 1, 0), this.targetOrientation);

    let mx = input.axis('KeyA', 'KeyD');
    let mz = input.axis('KeyS', 'KeyW');
    let my = (input.down('Space') ? 1 : 0) - (input.down('ShiftLeft') || input.down('ShiftRight') ? 1 : 0);
    if (input.down('ArrowUp')) mz += 1;
    if (input.down('ArrowDown')) mz -= 1;
    if (input.down('ArrowLeft')) mx -= 1;
    if (input.down('ArrowRight')) mx += 1;

    const sprinting = input.down('ControlLeft') || input.down('KeyF');
    this.boost = damp(this.boost, sprinting ? 8 : 1, 6, dt);

    const move = v3();
    v3addScaled(move, move, fwd, mz);
    v3addScaled(move, move, rgt, mx);
    v3addScaled(move, move, upv, my);
    const ml = v3len(move);
    if (ml > 0) {
      v3scale(move, move, 1 / ml);
      const s = this.speed * this.boost * speedScale;
      v3addScaled(this.targetPosition, this.targetPosition, move, s * dt);
    }

    // Field of view widens slightly with speed for a sense of velocity.
    const fovBoost = opts.fovBoost === false ? 0 : Math.min(0.16, (this.boost - 1) * 0.02 + ml * 0.03);
    this.targetFov = (opts.baseFov || 62) * DEG * (1 + fovBoost);
  }

  orbitUpdate(input, dt) {
    const o = this.orbit;
    if (input.buttons.has(0) || input.pointerLocked || input.isTouch) {
      o.yaw -= input.mouseDX * 0.005;
      o.pitch = clamp(o.pitch - input.mouseDY * 0.005, -1.5, 1.5);
    }
    if (input.wheel) o.distance = clamp(o.distance * Math.exp(input.wheel * 0.16), o.minDistance, o.maxDistance);
    if (input.pinch) o.distance = clamp(o.distance * Math.exp(-input.pinch * 0.5), o.minDistance, o.maxDistance);
    const cp = Math.cos(o.pitch), sp = Math.sin(o.pitch);
    v3set(this.targetPosition,
      o.center[0] + Math.sin(o.yaw) * cp * o.distance,
      o.center[1] + sp * o.distance,
      o.center[2] + Math.cos(o.yaw) * cp * o.distance);
    this.lookAt(o.center, v3(0, 1, 0));
  }

  shake(amount) { this._shake = Math.max(this._shake, amount); }

  update(dt, aspect) {
    this.aspect = aspect;
    const k = this.smoothing;
    this.position[0] = damp(this.position[0], this.targetPosition[0], k, dt);
    this.position[1] = damp(this.position[1], this.targetPosition[1], k, dt);
    this.position[2] = damp(this.position[2], this.targetPosition[2], k, dt);
    qslerp(this.orientation, this.orientation, this.targetOrientation, 1 - Math.exp(-k * dt));
    this.fov = damp(this.fov, this.targetFov, 6, dt);

    if (this._shake > 1e-4) {
      this._shakeSeed += dt * 37;
      const s = this._shake;
      const n = (a) => Math.sin(a * 12.9898) * 43758.5453 % 1;
      qmul(this.orientation, this.orientation,
        qsetAxisAngle(this._tmpQ, this._axisX, n(this._shakeSeed) * s * 0.01));
      qmul(this.orientation, this.orientation,
        qsetAxisAngle(this._tmpQ, this._axisY, n(this._shakeSeed + 7.3) * s * 0.01));
      this._shake = damp(this._shake, 0, 4, dt);
    }

    v3transformQuat(this.forward, v3(0, 0, -1), this.orientation);
    v3transformQuat(this.right, v3(1, 0, 0), this.orientation);
    v3transformQuat(this.up, v3(0, 1, 0), this.orientation);

    m4view(this.view, this.position, this.orientation);
    m4perspectiveReverseInfinite(this.proj, this.fov, aspect, this.near);
    m4mul(this.viewProj, this.proj, this.view);
    m4invert(this.invViewProj, this.viewProj);
    extractFrustum(this.frustum, this.viewProj);
  }

  // Camera-relative view matrix: translation removed, for rendering geometry
  // that is already expressed relative to the eye (avoids float32 blowup).
  viewNoTranslation(out) {
    m4view(out, v3(0, 0, 0), this.orientation);
    return out;
  }

  // World ray for a normalised device coordinate.
  rayFromNDC(ndcX, ndcY, outOrigin, outDir) {
    const tanF = Math.tan(this.fov * 0.5);
    const dx = ndcX * tanF * this.aspect;
    const dy = ndcY * tanF;
    v3set(outDir,
      this.right[0] * dx + this.up[0] * dy + this.forward[0],
      this.right[1] * dx + this.up[1] * dy + this.forward[1],
      this.right[2] * dx + this.up[2] * dy + this.forward[2]);
    v3norm(outDir, outDir);
    v3copy(outOrigin, this.position);
  }
}
