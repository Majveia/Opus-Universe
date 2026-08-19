// Turns a halo catalogue into a population of galaxies.
//
// Nothing here is placed by hand. A galaxy's stellar mass comes from its halo
// mass by abundance matching, its disk size from the halo's measured spin, its
// orientation from the halo's angular momentum vector, and its satellites from
// a subhalo mass function distributed on an NFW profile. Change the cosmology
// and the whole population changes with it.
//
// References:
//   Mo, Mao & White (1998), MNRAS 295, 319       - disk sizes from halo spin
//   Navarro, Frenk & White (1997), ApJ 490, 493  - halo density profile
//   Dutton & Maccio (2014), MNRAS 441, 3359      - concentration-mass relation
//   Giocoli et al. (2010), MNRAS 404, 502        - subhalo mass function
//   Behroozi et al. (2013), ApJ 770, 57          - stellar mass-halo mass

import { RNG, hash3 } from '../core/rng.js';
import { stellarMassFromHalo } from './halos.js';
import { blackbodyRGB } from '../render/color.js';

export const GALAXY_TYPE = { ELLIPTICAL: 0, SPIRAL: 1, IRREGULAR: 2 };

// Halo concentration from mass and epoch (Dutton & Maccio 2014, NFW, 200c).
function concentration(mHalo, z) {
  const a = 0.520 + (0.905 - 0.520) * Math.exp(-0.617 * Math.pow(z, 1.21));
  const b = -0.101 + 0.026 * z;
  return Math.pow(10, a + b * (Math.log10(mHalo) - 12));
}

// Inverse of the NFW cumulative mass profile, sampled by rejection on the
// dimensionless mass fraction m(x) = ln(1+x) - x/(1+x).
function sampleNFWRadius(rng, c) {
  const mTot = Math.log(1 + c) - c / (1 + c);
  const target = rng.f() * mTot;
  let lo = 0, hi = c;
  for (let i = 0; i < 40; i++) {
    const mid = 0.5 * (lo + hi);
    const m = Math.log(1 + mid) - mid / (1 + mid);
    if (m < target) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi) / c;      // as a fraction of the virial radius
}

function orthonormalBasis(n) {
  const a = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  let u = [
    a[1] * n[2] - a[2] * n[1],
    a[2] * n[0] - a[0] * n[2],
    a[0] * n[1] - a[1] * n[0],
  ];
  const ul = Math.hypot(u[0], u[1], u[2]) || 1;
  u = [u[0] / ul, u[1] / ul, u[2] / ul];
  const v = [
    n[1] * u[2] - n[2] * u[1],
    n[2] * u[0] - n[0] * u[2],
    n[0] * u[1] - n[1] * u[0],
  ];
  return [u, v];
}

/**
 * Builds the galaxy population for a halo catalogue.
 * @param {Array} halos      from HaloFinder
 * @param {Cosmology} cosmo
 * @param {object} o
 */
export function populate(halos, cosmo, { a = 1, boxSize = 150, seed = 1, maxGalaxies = 6000, minStellarMass = 3e7 } = {}) {
  const z = 1 / a - 1;
  const galaxies = [];

  for (let hi = 0; hi < halos.length; hi++) {
    const h = halos[hi];
    if (galaxies.length >= maxGalaxies) break;

    const mStar = stellarMassFromHalo(h.mass, a);
    if (mStar < minStellarMass) continue;

    // Deterministic per-halo randomness: the same halo always yields the same
    // galaxy, however many times the catalogue is rebuilt.
    const rng = new RNG(hash3(h.cell[0], h.cell[1], h.cell[2]) ^ seed);

    const c = concentration(h.mass, z);
    const central = makeGalaxy({
      rng, cosmo, a, z,
      mHalo: h.mass, mStar,
      rVir: h.rVir, spin: h.spin, axis: h.spinAxis,
      pos: [h.pos[0] * boxSize, h.pos[1] * boxSize, h.pos[2] * boxSize],
      vel: h.vel, sigmaV: h.sigmaV,
      isSatellite: false,
    });
    galaxies.push(central);

    /* -- satellites ---------------------------------------------------- */
    // The subhalo mass function is close to dN/d(m/M) ~ 0.3 (m/M)^-1.9 down to
    // the resolution limit; drawing from it puts the right number of companions
    // around a cluster and essentially none around a dwarf.
    const fMin = 0.005, fMax = 0.4;
    const slope = -1.9;
    const norm = 0.13;
    const expected = norm / (slope + 1) * (Math.pow(fMax, slope + 1) - Math.pow(fMin, slope + 1));
    let nSat = Math.floor(expected);
    if (rng.f() < expected - nSat) nSat++;
    nSat = Math.min(nSat, 40);

    for (let s = 0; s < nSat && galaxies.length < maxGalaxies; s++) {
      const f = rng.powerLaw(fMin, fMax, slope);
      const mSub = h.mass * f;
      const mStarSub = stellarMassFromHalo(mSub, a);
      if (mStarSub < minStellarMass) continue;

      const rFrac = sampleNFWRadius(rng, c);
      const dir = rng.onSphere([0, 0, 0]);
      const r = rFrac * h.rVir;
      const pos = [
        h.pos[0] * boxSize + dir[0] * r,
        h.pos[1] * boxSize + dir[1] * r,
        h.pos[2] * boxSize + dir[2] * r,
      ];
      // Satellites orbit; give them a velocity of order the halo dispersion.
      const vel = [
        h.vel[0] + rng.normal() * h.sigmaV,
        h.vel[1] + rng.normal() * h.sigmaV,
        h.vel[2] + rng.normal() * h.sigmaV,
      ];
      const axis = rng.onSphere([0, 0, 0]);
      galaxies.push(makeGalaxy({
        rng, cosmo, a, z,
        mHalo: mSub, mStar: mStarSub,
        rVir: h.rVir * Math.pow(f, 1 / 3),
        // Satellites are tidally stripped and preferentially quenched, which is
        // why cluster cores are full of red spheroids rather than blue disks.
        spin: Math.max(0.01, h.spin * 0.7 * (0.5 + rng.f())),
        axis, pos, vel, sigmaV: h.sigmaV * 0.35,
        isSatellite: true,
        // Environmental quenching needs two things: a host massive enough to
        // hold a hot gaseous halo, and an orbit deep enough inside it for ram
        // pressure and tidal stripping to act. Both together reproduce the
        // morphology-density relation, where cluster cores are dominated by
        // early types while the field is not.
        quenchBoost: smooth(12.6, 14.2, Math.log10(h.mass)) * Math.max(0, 1 - rFrac * 1.7),
      }));
    }
  }

  galaxies.sort((p, q) => q.luminosity - p.luminosity);
  return galaxies;
}

function makeGalaxy({ rng, cosmo, a, z, mHalo, mStar, rVir, spin, axis, pos, vel, sigmaV, isSatellite, quenchBoost = 0 }) {
  const logMStar = Math.log10(mStar);

  // Morphology. Massive galaxies are spheroids because their mergers destroyed
  // the disk; low-spin haloes cannot support a disk either; satellites deep in
  // a cluster have been stripped and quenched.
  // Mass is the dominant driver of morphology; spin and environment modulate
  // it. Calibrated so the field is spiral-dominated while cluster cores invert
  // to early types, which is the observed morphology-density relation.
  const massDrive = smooth(10.4, 11.5, logMStar);
  const spinDrive = 1 - smooth(0.015, 0.075, spin);
  const pEll = Math.min(0.97, massDrive * 0.88 + spinDrive * 0.14 + quenchBoost * 0.72);
  let type;
  if (rng.f() < pEll) type = GALAXY_TYPE.ELLIPTICAL;
  else if (logMStar < 8.7) type = GALAXY_TYPE.IRREGULAR;
  else type = GALAXY_TYPE.SPIRAL;

  // Disk size. Pure angular momentum conservation (Mo, Mao & White) gives
  // R_d = (lambda / sqrt(2)) R_vir, but that assumes the disk keeps all of its
  // halo's specific angular momentum - which fails badly for cluster-scale
  // haloes, where most of the gas never cools and no disk of that size exists.
  // So the absolute scale is anchored to the observed size-mass relation and
  // spin is kept as the modulation around it, which is the part it really does
  // control: high-spin haloes host large diffuse disks, low-spin ones compact.
  let radius;
  if (type === GALAXY_TYPE.ELLIPTICAL) {
    radius = 0.0025 * Math.pow(mStar / 1e10, 0.56);
  } else {
    const rObserved = 0.0018 * Math.pow(Math.max(mStar, 1e6) / 5e10, 0.22);
    radius = rObserved * Math.pow(Math.max(spin, 0.005) / 0.035, 0.6);
  }
  radius = Math.max(0.0004, Math.min(0.06, radius));   // Mpc/h: 0.4 kpc to 60 kpc

  // Star formation: a declining main sequence, shut off in spheroids.
  const sfrMS = Math.pow(10, -0.5) * Math.pow(mStar / 1e10, 0.7) * Math.pow(1 + z, 2.4);
  const quenched = type === GALAXY_TYPE.ELLIPTICAL ? 0.02 : (isSatellite ? 0.45 : 1.0);
  const sfr = sfrMS * quenched;
  const specificSFR = sfr / mStar;

  // The integrated colour is a mix of an old population and whatever is being
  // formed now, weighted by the specific star formation rate.
  const young = Math.min(1, specificSFR * 3e9);
  const tOld = 4300 - 400 * Math.min(1, (logMStar - 9) / 2);   // more massive = redder
  const tYoung = 12000;
  const cOld = blackbodyRGB(tOld);
  const cYoung = blackbodyRGB(tYoung);
  const w = type === GALAXY_TYPE.ELLIPTICAL ? 0.03 : 0.12 + 0.45 * young;
  const color = [
    cOld[0] * (1 - w) + cYoung[0] * w,
    cOld[1] * (1 - w) + cYoung[1] * w,
    cOld[2] * (1 - w) + cYoung[2] * w,
  ];

  // V-band mass-to-light ratio: old populations are dimmer per unit mass.
  const ml = type === GALAXY_TYPE.ELLIPTICAL ? 4.0 : 2.2 - 1.0 * young;
  const luminosity = mStar / ml;                    // solar luminosities

  const [u, v] = orthonormalBasis(axis);

  // Spiral structure. The pitch angle of the arms correlates with the mass
  // concentration: flocculent late types are loosely wound, grand-design early
  // types tightly wound.
  const armCount = type === GALAXY_TYPE.SPIRAL ? (rng.f() < 0.62 ? 2 : (rng.f() < 0.6 ? 3 : 4)) : 0;
  const pitchDeg = 8 + 22 * (1 - massDrive) * (0.6 + 0.8 * rng.f());
  const barStrength = type === GALAXY_TYPE.SPIRAL ? (rng.f() < 0.55 ? rng.range(0.2, 0.9) : 0) : 0;

  return {
    pos, vel, axis, u, v,
    radius,
    type,
    mStar, mHalo, luminosity, color,
    sfr, specificSFR, young,
    armCount,
    pitch: pitchDeg * Math.PI / 180,
    armStrength: type === GALAXY_TYPE.SPIRAL ? 0.55 + 0.45 * rng.f() : 0,
    dust: type === GALAXY_TYPE.SPIRAL ? 0.25 + 0.5 * young : 0.04,
    bar: barStrength,
    bulgeFraction: type === GALAXY_TYPE.ELLIPTICAL ? 1.0 : Math.min(0.7, 0.06 + 0.55 * massDrive),
    axisRatio: type === GALAXY_TYPE.ELLIPTICAL ? rng.range(0.55, 1.0) : 1.0,
    seed: rng.u32(),
    isSatellite,
    sigmaV,
  };
}

function smooth(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

// Packs the population into an RGBA32F texture the renderer can index by
// instance: eight texels per galaxy, one row each.
export const GALAXY_TEXELS = 8;

export function packGalaxies(galaxies, boxSize) {
  const n = galaxies.length;
  const data = new Float32Array(GALAXY_TEXELS * Math.max(1, n) * 4);
  for (let i = 0; i < n; i++) {
    const g = galaxies[i];
    const o = i * GALAXY_TEXELS * 4;
    data[o + 0] = g.pos[0]; data[o + 1] = g.pos[1]; data[o + 2] = g.pos[2]; data[o + 3] = g.radius;
    data[o + 4] = g.axis[0]; data[o + 5] = g.axis[1]; data[o + 6] = g.axis[2]; data[o + 7] = g.type;
    data[o + 8] = g.u[0]; data[o + 9] = g.u[1]; data[o + 10] = g.u[2]; data[o + 11] = g.axisRatio;
    data[o + 12] = g.v[0]; data[o + 13] = g.v[1]; data[o + 14] = g.v[2]; data[o + 15] = g.bulgeFraction;
    data[o + 16] = g.color[0]; data[o + 17] = g.color[1]; data[o + 18] = g.color[2];
    data[o + 19] = Math.log10(Math.max(1, g.luminosity));
    data[o + 20] = g.armCount; data[o + 21] = g.pitch; data[o + 22] = g.armStrength; data[o + 23] = g.dust;
    data[o + 24] = g.bar; data[o + 25] = g.young; data[o + 26] = (g.seed >>> 0) % 65536;
    data[o + 27] = Math.log10(Math.max(1, g.mStar));
    data[o + 28] = g.vel[0]; data[o + 29] = g.vel[1]; data[o + 30] = g.vel[2]; data[o + 31] = g.sigmaV;
  }
  return data;
}

// Summary statistics for the interface and for validation against observation.
export function populationStats(galaxies, boxSize) {
  const V = boxSize ** 3;
  const byType = [0, 0, 0];
  let mStarTotal = 0;
  for (const g of galaxies) { byType[g.type]++; mStarTotal += g.mStar; }
  return {
    count: galaxies.length,
    numberDensity: galaxies.length / V,
    ellipticals: byType[0], spirals: byType[1], irregulars: byType[2],
    stellarMassDensity: mStarTotal / V,
    brightest: galaxies.length ? galaxies[0] : null,
  };
}
