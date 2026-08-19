// Minimal, allocation-conscious linear algebra for the simulation and renderer.
// Matrices are column-major Float32Array(16), matching WebGL's expectations.

export const TAU = Math.PI * 2;
export const DEG = Math.PI / 180;

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const mix = (a, b, t) => a + (b - a) * t;
export const smoothstep = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};
// Frame-rate independent exponential approach: `rate` is the fraction closed per second.
export const damp = (a, b, rate, dt) => mix(a, b, 1 - Math.exp(-rate * dt));

/* ---------------------------------------------------------------- vec3 --- */

export const v3 = (x = 0, y = 0, z = 0) => new Float32Array([x, y, z]);
export const v3set = (o, x, y, z) => { o[0] = x; o[1] = y; o[2] = z; return o; };
export const v3copy = (o, a) => { o[0] = a[0]; o[1] = a[1]; o[2] = a[2]; return o; };
export const v3add = (o, a, b) => { o[0] = a[0] + b[0]; o[1] = a[1] + b[1]; o[2] = a[2] + b[2]; return o; };
export const v3sub = (o, a, b) => { o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2]; return o; };
export const v3scale = (o, a, s) => { o[0] = a[0] * s; o[1] = a[1] * s; o[2] = a[2] * s; return o; };
export const v3addScaled = (o, a, b, s) => { o[0] = a[0] + b[0] * s; o[1] = a[1] + b[1] * s; o[2] = a[2] + b[2] * s; return o; };
export const v3dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const v3len = (a) => Math.hypot(a[0], a[1], a[2]);
export const v3dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export const v3cross = (o, a, b) => {
  const ax = a[0], ay = a[1], az = a[2], bx = b[0], by = b[1], bz = b[2];
  o[0] = ay * bz - az * by; o[1] = az * bx - ax * bz; o[2] = ax * by - ay * bx;
  return o;
};
export const v3norm = (o, a) => {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (l > 0) { o[0] = a[0] / l; o[1] = a[1] / l; o[2] = a[2] / l; } else { o[0] = o[1] = o[2] = 0; }
  return o;
};
export const v3lerp = (o, a, b, t) => {
  o[0] = a[0] + (b[0] - a[0]) * t; o[1] = a[1] + (b[1] - a[1]) * t; o[2] = a[2] + (b[2] - a[2]) * t;
  return o;
};
export const v3transformQuat = (o, a, q) => {
  // o = a + 2 * cross(q.xyz, cross(q.xyz, a) + q.w * a)
  const x = a[0], y = a[1], z = a[2];
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  let ux = qy * z - qz * y + qw * x;
  let uy = qz * x - qx * z + qw * y;
  let uz = qx * y - qy * x + qw * z;
  o[0] = x + 2 * (qy * uz - qz * uy);
  o[1] = y + 2 * (qz * ux - qx * uz);
  o[2] = z + 2 * (qx * uy - qy * ux);
  return o;
};

/* ---------------------------------------------------------------- quat --- */

export const quat = () => new Float32Array([0, 0, 0, 1]);
export const qcopy = (o, a) => { o[0] = a[0]; o[1] = a[1]; o[2] = a[2]; o[3] = a[3]; return o; };
export const qidentity = (o) => { o[0] = o[1] = o[2] = 0; o[3] = 1; return o; };
export const qmul = (o, a, b) => {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  o[0] = ax * bw + aw * bx + ay * bz - az * by;
  o[1] = ay * bw + aw * by + az * bx - ax * bz;
  o[2] = az * bw + aw * bz + ax * by - ay * bx;
  o[3] = aw * bw - ax * bx - ay * by - az * bz;
  return o;
};
export const qsetAxisAngle = (o, axis, rad) => {
  const h = rad * 0.5, s = Math.sin(h);
  o[0] = axis[0] * s; o[1] = axis[1] * s; o[2] = axis[2] * s; o[3] = Math.cos(h);
  return o;
};
export const qnorm = (o, a) => {
  const l = Math.hypot(a[0], a[1], a[2], a[3]) || 1;
  o[0] = a[0] / l; o[1] = a[1] / l; o[2] = a[2] / l; o[3] = a[3] / l;
  return o;
};
export const qconj = (o, a) => { o[0] = -a[0]; o[1] = -a[1]; o[2] = -a[2]; o[3] = a[3]; return o; };
export const qslerp = (o, a, b, t) => {
  let ax = a[0], ay = a[1], az = a[2], aw = a[3];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) { cos = -cos; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  let s0, s1;
  if (1 - cos > 1e-6) {
    const omega = Math.acos(cos), sin = Math.sin(omega);
    s0 = Math.sin((1 - t) * omega) / sin;
    s1 = Math.sin(t * omega) / sin;
  } else { s0 = 1 - t; s1 = t; }
  o[0] = s0 * ax + s1 * bx; o[1] = s0 * ay + s1 * by;
  o[2] = s0 * az + s1 * bz; o[3] = s0 * aw + s1 * bw;
  return qnorm(o, o);
};
// Shortest-arc rotation carrying unit vector `from` onto unit vector `to`.
export const qfromTo = (o, from, to) => {
  const d = v3dot(from, to);
  if (d > 0.999999) return qidentity(o);
  if (d < -0.999999) {
    // Antiparallel: rotate by pi about any axis orthogonal to `from`.
    let ax = v3(1, 0, 0);
    if (Math.abs(from[0]) > 0.9) ax = v3(0, 1, 0);
    const c = v3cross(v3(), from, ax);
    v3norm(c, c);
    o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = 0;
    return o;
  }
  const c = v3cross(v3(), from, to);
  o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; o[3] = 1 + d;
  return qnorm(o, o);
};

/* ---------------------------------------------------------------- mat4 --- */

export const m4 = () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
export const m4identity = (o) => {
  o.fill(0); o[0] = o[5] = o[10] = o[15] = 1; return o;
};
export const m4mul = (o, a, b) => {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    o[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    o[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    o[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    o[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return o;
};
export const m4fromQuat = (o, q) => {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  o[0] = 1 - (yy + zz); o[1] = xy + wz;       o[2] = xz - wy;       o[3] = 0;
  o[4] = xy - wz;       o[5] = 1 - (xx + zz); o[6] = yz + wx;       o[7] = 0;
  o[8] = xz + wy;       o[9] = yz - wx;       o[10] = 1 - (xx + yy); o[11] = 0;
  o[12] = 0; o[13] = 0; o[14] = 0; o[15] = 1;
  return o;
};
export const m4compose = (o, pos, q, scale = 1) => {
  m4fromQuat(o, q);
  for (let i = 0; i < 12; i++) o[i] *= scale;
  o[12] = pos[0]; o[13] = pos[1]; o[14] = pos[2];
  return o;
};
// View matrix from a camera position and orientation (inverse of the rigid transform).
export const m4view = (o, pos, q) => {
  const inv = qconj(quat(), q);
  m4fromQuat(o, inv);
  const t = v3transformQuat(v3(), pos, inv);
  o[12] = -t[0]; o[13] = -t[1]; o[14] = -t[2];
  return o;
};
// Reversed-Z infinite perspective: maps near->1, infinity->0.
// Reversed Z keeps float depth precision uniform across an enormous view range.
export const m4perspectiveReverseInfinite = (o, fovY, aspect, near) => {
  const f = 1 / Math.tan(fovY * 0.5);
  o.fill(0);
  o[0] = f / aspect; o[5] = f;
  o[10] = 0; o[11] = -1;
  o[14] = near;
  return o;
};
export const m4perspective = (o, fovY, aspect, near, far) => {
  const f = 1 / Math.tan(fovY * 0.5);
  o.fill(0);
  o[0] = f / aspect; o[5] = f;
  o[10] = (far + near) / (near - far); o[11] = -1;
  o[14] = (2 * far * near) / (near - far);
  return o;
};
export const m4ortho = (o, l, r, b, t, n, f) => {
  o.fill(0);
  o[0] = 2 / (r - l); o[5] = 2 / (t - b); o[10] = -2 / (f - n); o[15] = 1;
  o[12] = -(r + l) / (r - l); o[13] = -(t + b) / (t - b); o[14] = -(f + n) / (f - n);
  return o;
};
export const m4invert = (o, m) => {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1 / det;
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return o;
};

/* ------------------------------------------------------------- frustum --- */

// Extracts the six clip planes from a view-projection matrix (Gribb & Hartmann).
// Planes are stored as [nx, ny, nz, d] with the interior on the positive side.
export const extractFrustum = (out, m) => {
  for (let i = 0; i < 6; i++) {
    const s = i & 1 ? -1 : 1;
    const r = i >> 1;
    const p = out[i] || (out[i] = new Float32Array(4));
    for (let c = 0; c < 4; c++) p[c] = m[c * 4 + 3] + s * m[c * 4 + r];
    const l = Math.hypot(p[0], p[1], p[2]) || 1;
    p[0] /= l; p[1] /= l; p[2] /= l; p[3] /= l;
  }
  return out;
};
export const sphereInFrustum = (planes, cx, cy, cz, r) => {
  for (let i = 0; i < 6; i++) {
    const p = planes[i];
    if (p[0] * cx + p[1] * cy + p[2] * cz + p[3] < -r) return false;
  }
  return true;
};
