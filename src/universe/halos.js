// Finds gravitationally bound structures in the simulated particle set.
//
// Uses friends-of-friends: two particles belong to the same halo if they lie
// within a linking length of each other, and membership is transitive. With the
// standard linking length of 0.2 mean interparticle separations, the groups
// this picks out enclose roughly the virial overdensity - and, importantly, the
// definition does not depend on any mesh, so it stays meaningful at whatever
// resolution the device can afford. What changes with resolution is only the
// smallest halo that can be resolved at all, which is reported alongside the
// catalogue rather than hidden.
//
// Each halo's bulk velocity, angular momentum and internal dispersion come from
// its own member particles, so a galaxy placed here inherits its spin axis from
// the tidal torques the simulation actually produced.
//
// References:
//   Davis, Efstathiou, Frenk & White (1985), ApJ 292, 371 - friends-of-friends
//   Bryan & Norman (1998), ApJ 495, 80                    - virial overdensity
//   Bullock et al. (2001), ApJ 555, 240                   - spin parameter
//   Behroozi, Wechsler & Conroy (2013), ApJ 770, 57       - stellar mass

import { atlasIndex } from '../sim/atlas.js';

export class HaloFinder {
  constructor(pm, cosmology) {
    this.pm = pm;
    this.cosmo = cosmology;
    this.n = pm.n;
    this.layout = pm.layout;
    this.boxSize = pm.boxSize;
    this.halos = [];
    const total = pm.particleCount;
    this.particleMass = cosmology.rhoMeanComoving * Math.pow(pm.boxSize, 3) / total;
    this._pos = new Float32Array(this.layout.width * this.layout.height * 4);
    this._vel = new Float32Array(this.layout.width * this.layout.height * 4);
    this._parent = new Int32Array(total);
    this._size = new Int32Array(total);
    this._atlasOf = new Int32Array(total);
    let w = 0;
    const n = this.n;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      this._atlasOf[w++] = atlasIndex(this.layout, x, y, z);
    }
  }

  _find(i) {
    const p = this._parent;
    let r = i;
    while (p[r] !== r) r = p[r];
    while (p[i] !== r) { const next = p[i]; p[i] = r; i = next; }
    return r;
  }

  _union(a, b) {
    let ra = this._find(a), rb = this._find(b);
    if (ra === rb) return;
    if (this._size[ra] < this._size[rb]) { const t = ra; ra = rb; rb = t; }
    this._parent[rb] = ra;
    this._size[ra] += this._size[rb];
  }

  /**
   * @param {object} o
   * @param {number} o.a              current scale factor
   * @param {number} o.linkingLength  in units of the mean interparticle separation
   * @param {number} o.minParticles   smallest group counted as a halo
   * @param {number} o.maxHalos
   */
  find({ a = 1, linkingLength = 0.2, minParticles = 20, maxHalos = 4000 } = {}) {
    const pm = this.pm;
    const total = pm.particleCount;
    const perSide = Math.round(Math.cbrt(total));
    const pos = this._pos, vel = this._vel;
    pm.posFBO[pm.curPos].readPixels(pos);
    pm.velFBO[pm.curVel].readPixels(vel);

    // Everything below works in box units, where the periodic volume is [0,1).
    const b = linkingLength / perSide;
    const b2 = b * b;
    const K = Math.max(1, Math.min(1024, Math.floor(1 / b)));   // hash cells per side
    const cell = 1 / K;

    const parent = this._parent, size = this._size, atlasOf = this._atlasOf;
    for (let i = 0; i < total; i++) { parent[i] = i; size[i] = 1; }

    /* -- 1. bucket particles into a hash grid of cells about one linking
           length across, so a neighbour search never leaves the 27 cells
           around a particle ------------------------------------------------ */
    const keys = new Int32Array(total);
    const counts = new Int32Array(K * K * K + 1);
    for (let i = 0; i < total; i++) {
      const ai = atlasOf[i];
      const cx = Math.min(K - 1, (pos[ai * 4] * K) | 0);
      const cy = Math.min(K - 1, (pos[ai * 4 + 1] * K) | 0);
      const cz = Math.min(K - 1, (pos[ai * 4 + 2] * K) | 0);
      const k = cx + K * (cy + K * cz);
      keys[i] = k;
      counts[k + 1]++;
    }
    for (let i = 0; i < K * K * K; i++) counts[i + 1] += counts[i];
    const order = new Int32Array(total);
    const cursor = Int32Array.from(counts.subarray(0, K * K * K));
    for (let i = 0; i < total; i++) order[cursor[keys[i]]++] = i;

    /* -- 2. link neighbours ------------------------------------------- */
    const wrapDelta = (d) => d - Math.round(d);
    for (let cz = 0; cz < K; cz++) {
      for (let cy = 0; cy < K; cy++) {
        for (let cx = 0; cx < K; cx++) {
          const k = cx + K * (cy + K * cz);
          const s0 = counts[k], e0 = counts[k + 1];
          if (s0 === e0) continue;
          for (let dz = 0; dz <= 1; dz++) {
            for (let dy = (dz === 0 ? 0 : -1); dy <= 1; dy++) {
              for (let dx = (dz === 0 && dy === 0 ? 0 : -1); dx <= 1; dx++) {
                const nx = (cx + dx + K) % K, ny = (cy + dy + K) % K, nz = (cz + dz + K) % K;
                const nk = nx + K * (ny + K * nz);
                if (nk < k && !(dx === 0 && dy === 0 && dz === 0)) {
                  // Each unordered pair of cells is visited once; skip the mirror.
                }
                const s1 = counts[nk], e1 = counts[nk + 1];
                if (s1 === e1) continue;
                const same = nk === k;
                for (let ii = s0; ii < e0; ii++) {
                  const pi = order[ii];
                  const ai = atlasOf[pi];
                  const px = pos[ai * 4], py = pos[ai * 4 + 1], pz = pos[ai * 4 + 2];
                  for (let jj = same ? ii + 1 : s1; jj < e1; jj++) {
                    const pj = order[jj];
                    const aj = atlasOf[pj];
                    const ddx = wrapDelta(pos[aj * 4] - px);
                    const ddy = wrapDelta(pos[aj * 4 + 1] - py);
                    const ddz = wrapDelta(pos[aj * 4 + 2] - pz);
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= b2) this._union(pi, pj);
                  }
                }
              }
            }
          }
        }
      }
    }

    /* -- 3. collect groups -------------------------------------------- */
    const groupOf = new Map();
    for (let i = 0; i < total; i++) {
      const r = this._find(i);
      if (size[r] < minParticles) continue;
      let g = groupOf.get(r);
      if (!g) { g = []; groupOf.set(r, g); }
      g.push(i);
    }

    const deltaVir = this.cosmo.deltaVir(a);
    const rhoMean = this.cosmo.rhoMeanComoving;
    const halos = [];

    for (const members of groupOf.values()) {
      const nMem = members.length;
      const mass = nMem * this.particleMass;

      // Centre of mass, computed relative to the first member so the periodic
      // wrap never splits a halo straddling a box face.
      const a0 = atlasOf[members[0]];
      const rx0 = pos[a0 * 4], ry0 = pos[a0 * 4 + 1], rz0 = pos[a0 * 4 + 2];
      let sx = 0, sy = 0, sz = 0, vx = 0, vy = 0, vz = 0;
      for (const m of members) {
        const ai = atlasOf[m];
        sx += wrapDelta(pos[ai * 4] - rx0);
        sy += wrapDelta(pos[ai * 4 + 1] - ry0);
        sz += wrapDelta(pos[ai * 4 + 2] - rz0);
        vx += vel[ai * 4]; vy += vel[ai * 4 + 1]; vz += vel[ai * 4 + 2];
      }
      sx /= nMem; sy /= nMem; sz /= nMem;
      vx /= nMem; vy /= nMem; vz /= nMem;
      const cxb = ((rx0 + sx) % 1 + 1) % 1;
      const cyb = ((ry0 + sy) % 1 + 1) % 1;
      const czb = ((rz0 + sz) % 1 + 1) % 1;

      // Angular momentum and dispersion about that centre.
      let lx = 0, ly = 0, lz = 0, v2 = 0;
      for (const m of members) {
        const ai = atlasOf[m];
        const dx = wrapDelta(pos[ai * 4] - cxb);
        const dy = wrapDelta(pos[ai * 4 + 1] - cyb);
        const dz = wrapDelta(pos[ai * 4 + 2] - czb);
        const ux = vel[ai * 4] - vx, uy = vel[ai * 4 + 1] - vy, uz = vel[ai * 4 + 2] - vz;
        lx += dy * uz - dz * uy;
        ly += dz * ux - dx * uz;
        lz += dx * uy - dy * ux;
        v2 += ux * ux + uy * uy + uz * uz;
      }
      const lLen = Math.hypot(lx, ly, lz) || 1e-30;
      const sigmaV = Math.sqrt(v2 / nMem / 3);

      // Virial radius from the mass and the epoch's virial overdensity.
      const rVir = Math.cbrt(3 * mass / (4 * Math.PI * deltaVir * rhoMean));

      // Bullock spin parameter: dimensionless, so the internal units cancel.
      const specificJ = lLen / nMem;                     // box units
      const vVir = sigmaV * Math.SQRT2 || 1e-12;
      const spin = Math.max(0.005, Math.min(0.3,
        specificJ / (Math.SQRT2 * vVir * (rVir / this.boxSize) + 1e-30)));

      const n = this.n;
      halos.push({
        pos: [cxb, cyb, czb],
        cell: [Math.min(n - 1, (cxb * n) | 0), Math.min(n - 1, (cyb * n) | 0), Math.min(n - 1, (czb * n) | 0)],
        rVir,
        mass,
        vel: [vx, vy, vz],
        spinAxis: [lx / lLen, ly / lLen, lz / lLen],
        spin,
        sigmaV,
        nParticles: nMem,
      });
    }

    halos.sort((h, g) => g.mass - h.mass);
    if (halos.length > maxHalos) halos.length = maxHalos;
    this.halos = halos;
    this.deltaVir = deltaVir;
    this.scaleFactor = a;
    this.minResolvedMass = minParticles * this.particleMass;
    this.linkingLengthMpc = b * this.boxSize;
    return halos;
  }

  // Measured halo mass function, in number per (Mpc/h)^3 per dex.
  massFunction(bins = 8) {
    const hs = this.halos.filter((h) => h.mass > 0);
    if (!hs.length) return [];
    const lo = Math.log10(Math.min(...hs.map((h) => h.mass)));
    const hi = Math.log10(Math.max(...hs.map((h) => h.mass))) + 1e-6;
    const V = this.boxSize ** 3;
    const width = (hi - lo) / bins;
    const out = [];
    for (let i = 0; i < bins; i++) {
      const a = lo + i * width, b = a + width;
      const count = hs.filter((h) => Math.log10(h.mass) >= a && Math.log10(h.mass) < b).length;
      out.push({ logM: (a + b) / 2, count, dndlogM: count / V / width });
    }
    return out;
  }
}

/* --------------------------------------------------- galaxies from haloes -- */

// Stellar mass from halo mass, following the double power law of Behroozi,
// Wechsler & Conroy (2013). Star formation is inefficient at both ends: small
// haloes lose their gas to feedback, large ones shock-heat it and quench.
export function stellarMassFromHalo(mHalo, a = 1) {
  const z = 1 / a - 1;
  const nu = Math.exp(-4 * a * a);
  const logM1 = 11.514 + nu * (-1.793 * (a - 1) - 0.251 * z);
  const logEps = -1.777 + nu * (-0.006 * (a - 1)) - 0.119 * (a - 1);
  const alpha = -1.412 + nu * (0.731 * (a - 1));
  const delta = 3.508 + nu * (2.608 * (a - 1) - 0.043 * z);
  const gamma = 0.316 + nu * (1.319 * (a - 1) + 0.279 * z);

  const f = (x) => -Math.log10(Math.pow(10, alpha * x) + 1) +
    delta * Math.pow(Math.log10(1 + Math.exp(x)), gamma) / (1 + Math.exp(Math.pow(10, -x)));

  const x = Math.log10(mHalo) - logM1;
  return Math.pow(10, logEps + logM1 + f(x) - f(0));
}
