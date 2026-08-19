// Finds gravitationally bound structures in the simulated density field.
//
// Uses spherical overdensity: locate peaks in the density grid, then grow a
// sphere around each until the mean enclosed density falls to the virial
// threshold predicted for this cosmology. Particles inside that sphere give the
// halo its bulk velocity, angular momentum and internal velocity dispersion -
// so a galaxy placed here inherits its spin axis from the actual tidal torques
// the simulation produced, rather than from a random number.
//
// References:
//   Bryan & Norman (1998), ApJ 495, 80          - virial overdensity
//   Bullock et al. (2001), ApJ 555, 240         - spin parameter
//   Behroozi, Wechsler & Conroy (2013), ApJ 770, 57 - stellar mass to halo mass

import { atlasIndex } from '../sim/atlas.js';

export class HaloFinder {
  constructor(pm, cosmology) {
    this.pm = pm;
    this.cosmo = cosmology;
    this.n = pm.n;
    this.layout = pm.layout;
    this.boxSize = pm.boxSize;
    this.halos = [];
    this._grid = new Float32Array(this.n ** 3);
  }

  // Pulls the atlas-ordered density field into a grid-ordered array.
  _readDensityGrid() {
    const flat = this.pm.readDensity();
    const n = this.n, g = this._grid, layout = this.layout, mean = this.pm.meanDensity;
    for (let z = 0; z < n; z++) {
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          g[x + n * (y + n * z)] = flat[atlasIndex(layout, x, y, z)] / mean;
        }
      }
    }
    return g;
  }

  // 3x3x3 periodic box smoothing, twice: enough to stop single-cell shot noise
  // from registering as a peak without erasing genuine substructure.
  _smooth(src, passes = 1) {
    const n = this.n;
    let a = src, b = new Float32Array(n ** 3);
    for (let p = 0; p < passes; p++) {
      for (let z = 0; z < n; z++) {
        const zm = ((z - 1) + n) % n, zp = (z + 1) % n;
        for (let y = 0; y < n; y++) {
          const ym = ((y - 1) + n) % n, yp = (y + 1) % n;
          for (let x = 0; x < n; x++) {
            const xm = ((x - 1) + n) % n, xp = (x + 1) % n;
            // Separable 1-4-1 kernel applied as one gather; the weights below
            // are the 3D product, normalised.
            let s = 0;
            for (const zz of [zm, z, zp]) {
              const wz = zz === z ? 4 : 1;
              for (const yy of [ym, y, yp]) {
                const wy = yy === y ? 4 : 1;
                s += wz * wy * (a[xm + n * (yy + n * zz)] + 4 * a[x + n * (yy + n * zz)] + a[xp + n * (yy + n * zz)]);
              }
            }
            b[x + n * (y + n * z)] = s / 216;
          }
        }
      }
      const t = a; a = b; b = (t === src ? new Float32Array(n ** 3) : t);
    }
    return a;
  }

  /**
   * @param {object} o
   * @param {number} o.a               current scale factor
   * @param {number} o.peakThreshold   minimum overdensity for a peak
   * @param {number} o.maxHalos
   * @param {boolean} o.withKinematics read particles back for spin and dispersion
   */
  find({ a = 1, peakThreshold = 20, maxHalos = 4000, withKinematics = true } = {}) {
    const n = this.n;
    const raw = this._readDensityGrid();
    const rho = this._smooth(Float32Array.from(raw), 1);

    const deltaVir = this.cosmo.deltaVir(a);
    const cellSize = this.boxSize / n;                 // Mpc/h
    const cellVolume = cellSize ** 3;
    const rhoMean = this.cosmo.rhoMeanComoving;        // Msun/h per (Mpc/h)^3
    const massPerUnitDensity = rhoMean * cellVolume;   // mass of a cell at rho/rho_bar = 1

    /* -- 1. local maxima above the threshold --------------------------- */
    const peaks = [];
    const at = (x, y, z) => rho[(((x % n) + n) % n) + n * ((((y % n) + n) % n) + n * (((z % n) + n) % n))];
    for (let z = 0; z < n; z++) {
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const v = rho[x + n * (y + n * z)];
          if (v < peakThreshold) continue;
          let isMax = true;
          for (let dz = -1; dz <= 1 && isMax; dz++)
            for (let dy = -1; dy <= 1 && isMax; dy++)
              for (let dx = -1; dx <= 1 && isMax; dx++) {
                if (!dx && !dy && !dz) continue;
                if (at(x + dx, y + dy, z + dz) > v) isMax = false;
              }
          if (isMax) peaks.push({ x, y, z, v });
        }
      }
    }
    peaks.sort((p, q) => q.v - p.v);

    /* -- 2. grow spheres to the virial overdensity ---------------------- */
    // Precomputed shell offsets, sorted by radius, so growth is a single sweep.
    const maxR = Math.min(n / 2 - 1, 24);
    const offsets = [];
    for (let dz = -maxR; dz <= maxR; dz++)
      for (let dy = -maxR; dy <= maxR; dy++)
        for (let dx = -maxR; dx <= maxR; dx++) {
          const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (r <= maxR) offsets.push({ dx, dy, dz, r });
        }
    offsets.sort((p, q) => p.r - q.r);

    const claimed = new Uint8Array(n ** 3);
    const halos = [];

    for (const p of peaks) {
      if (halos.length >= maxHalos) break;
      if (claimed[p.x + n * (p.y + n * p.z)]) continue;

      let mass = 0, volume = 0;
      let bestR = 0, bestMass = 0;
      let cx = 0, cy = 0, cz = 0;
      let i = 0;
      // Sweep outward; the virial radius is the largest radius at which the
      // mean enclosed overdensity still exceeds delta_vir.
      while (i < offsets.length) {
        const r = offsets[i].r;
        while (i < offsets.length && offsets[i].r === r) {
          const o = offsets[i];
          const d = at(p.x + o.dx, p.y + o.dy, p.z + o.dz);
          mass += d;
          cx += d * o.dx; cy += d * o.dy; cz += d * o.dz;
          volume += 1;
          i++;
        }
        if (volume > 4) {
          const meanEnclosed = mass / volume;
          if (meanEnclosed >= deltaVir) { bestR = r; bestMass = mass; }
          else if (bestR > 0) break;
          else if (r > 3) break;               // never reached the threshold
        }
      }
      if (bestR < 0.9 || bestMass <= 0) continue;

      const comX = p.x + cx / mass, comY = p.y + cy / mass, comZ = p.z + cz / mass;

      // Mark the volume as taken so substructure is not counted twice.
      const rad = Math.ceil(bestR);
      for (let dz = -rad; dz <= rad; dz++)
        for (let dy = -rad; dy <= rad; dy++)
          for (let dx = -rad; dx <= rad; dx++) {
            if (dx * dx + dy * dy + dz * dz > bestR * bestR) continue;
            const xx = (((p.x + dx) % n) + n) % n, yy = (((p.y + dy) % n) + n) % n, zz = (((p.z + dz) % n) + n) % n;
            claimed[xx + n * (yy + n * zz)] = 1;
          }

      halos.push({
        // Position in box units [0,1)
        pos: [((comX / n) % 1 + 1) % 1, ((comY / n) % 1 + 1) % 1, ((comZ / n) % 1 + 1) % 1],
        cell: [p.x, p.y, p.z],
        rVir: bestR * cellSize,                        // Mpc/h
        mass: bestMass * massPerUnitDensity,           // Msun/h
        peak: p.v,
        vel: [0, 0, 0],
        spinAxis: [0, 1, 0],
        spin: 0.035,
        sigmaV: 0,
        nParticles: 0,
      });
    }

    halos.sort((h, g) => g.mass - h.mass);
    this.halos = halos;
    if (withKinematics && halos.length) this._measureKinematics(halos);
    this.deltaVir = deltaVir;
    this.scaleFactor = a;
    return halos;
  }

  // Reads particles back once and assigns them to haloes through a uniform
  // spatial hash, then measures each halo's bulk motion and angular momentum.
  _measureKinematics(halos) {
    const pm = this.pm, n = this.n, layout = this.layout;
    const posFlat = new Float32Array(layout.width * layout.height * 4);
    const velFlat = new Float32Array(layout.width * layout.height * 4);
    pm.posFBO[pm.curPos].readPixels(posFlat);
    pm.velFBO[pm.curVel].readPixels(velFlat);

    // Bucket particles by cell so each halo only scans its own neighbourhood.
    const nCells = n ** 3;
    const counts = new Int32Array(nCells + 1);
    const total = pm.particleCount;
    const cellOf = new Int32Array(total);
    const atlasOf = new Int32Array(total);

    let w = 0;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const ai = atlasIndex(layout, x, y, z);
      atlasOf[w] = ai;
      const px = posFlat[ai * 4], py = posFlat[ai * 4 + 1], pz = posFlat[ai * 4 + 2];
      const ci = Math.min(n - 1, Math.floor(px * n)) + n * (Math.min(n - 1, Math.floor(py * n)) + n * Math.min(n - 1, Math.floor(pz * n)));
      cellOf[w] = ci;
      counts[ci + 1]++;
      w++;
    }
    for (let i = 0; i < nCells; i++) counts[i + 1] += counts[i];
    const order = new Int32Array(total);
    const cursor = Int32Array.from(counts.subarray(0, nCells));
    for (let i = 0; i < total; i++) order[cursor[cellOf[i]]++] = i;

    const cellSize = this.boxSize / n;
    const wrapDelta = (d) => d - Math.round(d);          // periodic separation in box units

    for (const h of halos) {
      const rBox = h.rVir / this.boxSize;
      const rCells = Math.ceil(h.rVir / cellSize) + 1;
      const c0 = h.cell;
      let m = 0, vx = 0, vy = 0, vz = 0;
      let lx = 0, ly = 0, lz = 0;
      let v2 = 0;
      const members = [];

      for (let dz = -rCells; dz <= rCells; dz++)
        for (let dy = -rCells; dy <= rCells; dy++)
          for (let dx = -rCells; dx <= rCells; dx++) {
            const xx = (((c0[0] + dx) % n) + n) % n, yy = (((c0[1] + dy) % n) + n) % n, zz = (((c0[2] + dz) % n) + n) % n;
            const ci = xx + n * (yy + n * zz);
            for (let k = counts[ci]; k < counts[ci + 1]; k++) {
              const pi = order[k];
              const ai = atlasOf[pi];
              const rx = wrapDelta(posFlat[ai * 4] - h.pos[0]);
              const ry = wrapDelta(posFlat[ai * 4 + 1] - h.pos[1]);
              const rz = wrapDelta(posFlat[ai * 4 + 2] - h.pos[2]);
              if (rx * rx + ry * ry + rz * rz > rBox * rBox) continue;
              members.push([rx, ry, rz, velFlat[ai * 4], velFlat[ai * 4 + 1], velFlat[ai * 4 + 2]]);
              m += 1;
              vx += velFlat[ai * 4]; vy += velFlat[ai * 4 + 1]; vz += velFlat[ai * 4 + 2];
            }
          }

      if (m < 8) { h.nParticles = m; continue; }
      vx /= m; vy /= m; vz /= m;
      for (const [rx, ry, rz, ux, uy, uz] of members) {
        const dx = ux - vx, dy = uy - vy, dz = uz - vz;
        lx += ry * dz - rz * dy;
        ly += rz * dx - rx * dz;
        lz += rx * dy - ry * dx;
        v2 += dx * dx + dy * dy + dz * dz;
      }
      const lLen = Math.hypot(lx, ly, lz) || 1;
      h.vel = [vx, vy, vz];
      h.spinAxis = [lx / lLen, ly / lLen, lz / lLen];
      h.sigmaV = Math.sqrt(v2 / m / 3);
      h.nParticles = m;

      // Bullock spin parameter, lambda' = J / (sqrt(2) M V_vir R_vir), computed
      // in the simulation's internal units - dimensionless, so units cancel.
      const specificJ = lLen / m;
      const vVir = h.sigmaV * Math.sqrt(3) || 1e-9;
      h.spin = Math.min(0.3, specificJ / (Math.SQRT2 * vVir * rBox + 1e-30));
    }
  }

  // Measured halo mass function, in number per (Mpc/h)^3 per dex, for
  // comparison against the Sheth-Tormen prediction.
  massFunction(bins = 8) {
    const hs = this.halos.filter((h) => h.mass > 0);
    if (!hs.length) return [];
    const lo = Math.log10(Math.min(...hs.map((h) => h.mass)));
    const hi = Math.log10(Math.max(...hs.map((h) => h.mass)));
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
  const logEps = -1.777 + nu * (-0.006 * (a - 1) - 0.000 * z) - 0.119 * (a - 1);
  const alpha = -1.412 + nu * (0.731 * (a - 1));
  const delta = 3.508 + nu * (2.608 * (a - 1) - 0.043 * z);
  const gamma = 0.316 + nu * (1.319 * (a - 1) + 0.279 * z);

  const f = (x) => -Math.log10(Math.pow(10, alpha * x) + 1) +
    delta * Math.pow(Math.log10(1 + Math.exp(x)), gamma) / (1 + Math.exp(Math.pow(10, -x)));

  const x = Math.log10(mHalo) - logM1;
  const logMstar = logEps + logM1 + f(x) - f(0);
  return Math.pow(10, logMstar);
}
