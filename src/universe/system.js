// Deterministic star systems.
//
// Given a 64-bit-ish seed, this builds a star and its planets from the same
// relations astronomers use: a Kroupa initial mass function for the star,
// main-sequence structure for its luminosity and temperature, a snow line set
// by that luminosity, Hill-stability spacing between orbits, and equilibrium
// temperatures that decide what each world turns out to be.
//
// Orbits are genuine Keplerian ellipses with all six elements, advanced by
// solving Kepler's equation, so periods obey T^2 = a^3 / M and inner planets
// really do lap outer ones.
//
// References:
//   Kroupa (2001), MNRAS 322, 231              - initial mass function
//   Chambers (1996) / Gladman (1993)           - mutual Hill spacing
//   Seager et al. (2007), ApJ 669, 1279        - rocky mass-radius relation
//   Chen & Kipping (2017), ApJ 834, 17         - the giant-planet branch

import { RNG } from '../core/rng.js';
import { mainSequence, spectralClass, blackbodyRGB } from '../render/color.js';

export const AU_IN_SOLAR_RADII = 215.032;
export const EARTH_RADII_PER_AU = 23454.8;
export const SIGMA_SB = 5.670374419e-8;

export const PLANET_TYPE = {
  LAVA: 0, ROCK: 1, DESERT: 2, TERRAN: 3, OCEAN: 4,
  ICE: 5, ICE_GIANT: 6, GAS_GIANT: 7,
};
export const PLANET_TYPE_NAME = [
  'lava world', 'rocky world', 'desert world', 'terran world', 'ocean world',
  'ice world', 'ice giant', 'gas giant',
];

// Main-sequence lifetime in Gyr. Massive stars burn through their fuel in a
// few million years; a red dwarf outlives the present age of the universe many
// times over.
export function mainSequenceLifetimeGyr(mass) {
  return 10 * Math.pow(mass, -2.5);
}

// Kroupa (2001) broken power law, sampled between 0.08 and 60 solar masses.
function sampleIMFBirth(rng) {
  // Segment weights are the integrals of each power-law piece.
  const segs = [
    { lo: 0.08, hi: 0.5, alpha: -1.3 },
    { lo: 0.5, hi: 60, alpha: -2.3 },
  ];
  const weight = segs.map((s) => {
    const a1 = s.alpha + 1;
    return (Math.pow(s.hi, a1) - Math.pow(s.lo, a1)) / a1;
  });
  // Continuity factor between the two segments.
  weight[1] *= Math.pow(0.5, -1.3) / Math.pow(0.5, -2.3);
  const total = weight[0] + weight[1];
  const u = rng.f() * total;
  const s = u < weight[0] ? segs[0] : segs[1];
  return rng.powerLaw(s.lo, s.hi, s.alpha);
}

// What you actually see is not the mass function stars are born with, but the
// one that survives. Weighting by the fraction of the galaxy's history a star
// of that mass can live through turns the birth IMF into the present-day
// population - which is why O and B stars are vanishingly rare on the sky even
// though star formation keeps making them.
function sampleIMF(rng, galaxyAgeGyr = 10) {
  for (let attempt = 0; attempt < 64; attempt++) {
    const m = sampleIMFBirth(rng);
    const survives = Math.min(1, mainSequenceLifetimeGyr(m) / galaxyAgeGyr);
    if (rng.f() < survives) return m;
  }
  return 0.3;
}

// Rocky worlds compress under their own gravity; giants are supported by
// degeneracy pressure and barely grow with mass at all.
function radiusFromMass(mEarth) {
  if (mEarth < 2) return Math.pow(mEarth, 0.27);
  if (mEarth < 130) return 0.808 * Math.pow(mEarth, 0.589);   // Chen & Kipping
  return 11.2 * Math.pow(mEarth / 318, -0.04);                 // ~1 Jupiter radius
}

function bondAlbedo(type, rng) {
  switch (type) {
    case PLANET_TYPE.ICE: return 0.55 + 0.2 * rng.f();
    case PLANET_TYPE.OCEAN: return 0.28 + 0.08 * rng.f();
    case PLANET_TYPE.TERRAN: return 0.29 + 0.08 * rng.f();
    case PLANET_TYPE.DESERT: return 0.22 + 0.1 * rng.f();
    case PLANET_TYPE.LAVA: return 0.06 + 0.06 * rng.f();
    case PLANET_TYPE.GAS_GIANT: return 0.32 + 0.15 * rng.f();
    case PLANET_TYPE.ICE_GIANT: return 0.28 + 0.12 * rng.f();
    default: return 0.12 + 0.15 * rng.f();
  }
}

export function generateSystem(seed, { galaxyContext = null } = {}) {
  const rng = new RNG(seed >>> 0);

  /* -- the star ------------------------------------------------------- */
  const mass = sampleIMF(rng);
  const star = mainSequence(mass);
  star.class = spectralClass(star.temperature);
  star.radiusAU = star.radius / AU_IN_SOLAR_RADII;
  star.color = blackbodyRGB(star.temperature);
  // Metallicity drives giant-planet occurrence; a metal-poor star rarely has one.
  star.metallicity = rng.normal() * 0.22 - 0.05;
  star.ageGyr = rng.range(0.2, Math.min(11, 10 * Math.pow(mass, -2.5) + 0.5));

  // A fraction of systems are binaries, rising steeply with primary mass.
  // Multiplicity rises steeply with primary mass: about a quarter of M dwarfs,
  // near half of solar types, and the great majority of O and B stars.
  const binaryProb = Math.min(0.80, 0.18 + 0.48 * Math.min(1, Math.log10(mass / 0.1) / 1.4));
  let companion = null;
  if (rng.f() < binaryProb) {
    const q = rng.range(0.12, 1.0);                  // mass ratio
    const cm = Math.max(0.08, mass * q);
    companion = mainSequence(cm);
    companion.class = spectralClass(companion.temperature);
    companion.radiusAU = companion.radius / AU_IN_SOLAR_RADII;
    companion.color = blackbodyRGB(companion.temperature);
    companion.separationAU = Math.pow(10, rng.range(-0.7, 2.6));
    companion.eccentricity = rng.f() * 0.6;
    companion.phase = rng.f() * Math.PI * 2;
  }

  /* -- where planets can exist ---------------------------------------- */
  // Dust sublimates inside this radius, so nothing accretes there.
  const rInner = Math.max(0.015, 0.034 * Math.sqrt(star.luminosity));
  const rSnow = 2.7 * Math.sqrt(star.luminosity);
  const rOuter = Math.min(120, 40 * Math.pow(star.luminosity, 0.25) * rng.range(0.6, 1.8));
  // Habitable zone from the stellar flux limits (runaway greenhouse to maximum
  // greenhouse), scaled by luminosity.
  const hzInner = 0.95 * Math.sqrt(star.luminosity);
  const hzOuter = 1.67 * Math.sqrt(star.luminosity);

  /* -- planets --------------------------------------------------------- */
  const nWanted = 1 + rng.int(9);
  const planets = [];
  let a = rInner * rng.range(1.05, 2.2);

  for (let i = 0; i < nWanted && a < rOuter; i++) {
    const beyondSnow = a > rSnow;
    // Giants form beyond the snow line where ice is available, and need metals.
    const giantChance = (beyondSnow ? 0.42 : 0.06) * Math.pow(10, star.metallicity * 1.2);
    const isGiant = rng.f() < giantChance;

    let mEarth;
    if (isGiant) mEarth = rng.powerLaw(9, 3000, -1.05);
    else mEarth = rng.powerLaw(0.04, 12, -0.9);

    const rEarth = radiusFromMass(mEarth);
    const radiusAU = rEarth / EARTH_RADII_PER_AU;

    const ecc = Math.min(0.75, Math.abs(rng.normal()) * 0.09 + (isGiant ? 0.01 : 0.02));
    const inc = Math.abs(rng.normal()) * 0.035;

    // Equilibrium temperature, before any greenhouse effect.
    let type = PLANET_TYPE.ROCK;
    let albedo = 0.3;
    let tEq = star.temperature * Math.sqrt(star.radiusAU / (2 * a)) * Math.pow(1 - albedo, 0.25);

    if (mEarth > 55) type = PLANET_TYPE.GAS_GIANT;
    else if (mEarth > 9) type = PLANET_TYPE.ICE_GIANT;
    else if (tEq > 1100) type = PLANET_TYPE.LAVA;
    else if (tEq < 165) type = PLANET_TYPE.ICE;
    else if (a >= hzInner && a <= hzOuter && mEarth > 0.35 && mEarth < 8) {
      type = rng.f() < 0.45 ? PLANET_TYPE.OCEAN : PLANET_TYPE.TERRAN;
    } else if (tEq > 320) type = PLANET_TYPE.DESERT;
    else type = PLANET_TYPE.ROCK;

    albedo = bondAlbedo(type, rng);
    tEq = star.temperature * Math.sqrt(star.radiusAU / (2 * a)) * Math.pow(1 - albedo, 0.25);
    // A thick atmosphere warms the surface above equilibrium.
    const greenhouse = type === PLANET_TYPE.GAS_GIANT || type === PLANET_TYPE.ICE_GIANT ? 1.0
      : type === PLANET_TYPE.OCEAN || type === PLANET_TYPE.TERRAN ? rng.range(1.03, 1.14)
      : type === PLANET_TYPE.LAVA ? rng.range(1.1, 1.6) : rng.range(1.0, 1.05);
    const tSurface = tEq * greenhouse;

    // Rings: a debris disk inside the Roche limit, favoured around giants with
    // an obliquity that keeps them from settling into the orbital plane.
    const obliquity = Math.abs(rng.normal()) * 0.42;
    const hasRings = (type === PLANET_TYPE.GAS_GIANT && rng.f() < 0.55)
      || (type === PLANET_TYPE.ICE_GIANT && rng.f() < 0.3);

    // Rotation: tidal locking wins close in.
    const tidalLockAU = 0.06 * Math.pow(star.mass, 1 / 3);
    const tidallyLocked = a < tidalLockAU;
    const rotationHours = tidallyLocked ? null : rng.range(6, 60);

    const planet = {
      index: i,
      semiMajorAU: a,
      eccentricity: ecc,
      inclination: inc,
      longitudeAscending: rng.f() * Math.PI * 2,
      argumentPeriapsis: rng.f() * Math.PI * 2,
      meanAnomaly0: rng.f() * Math.PI * 2,
      periodYears: Math.sqrt(a * a * a / (star.mass + (companion ? companion.mass * 0 : 0))),
      massEarth: mEarth,
      radiusEarth: rEarth,
      radiusAU,
      type,
      typeName: PLANET_TYPE_NAME[type],
      albedo,
      tEq, tSurface,
      obliquity,
      rotationHours,
      tidallyLocked,
      hasRings,
      ringInner: hasRings ? radiusAU * rng.range(1.4, 1.8) : 0,
      ringOuter: hasRings ? radiusAU * rng.range(2.1, 3.4) : 0,
      ringOpacity: hasRings ? rng.range(0.35, 0.95) : 0,
      seed: rng.u32(),
      moons: [],
      habitable: (type === PLANET_TYPE.OCEAN || type === PLANET_TYPE.TERRAN)
        && tSurface > 255 && tSurface < 320,
    };

    /* -- moons --------------------------------------------------------- */
    // Hill radius sets how far a satellite can orbit before the star strips it.
    const hill = a * Math.pow(mEarth / (3 * 333000 * star.mass), 1 / 3) * (1 - ecc);
    const nMoons = Math.min(9, Math.floor(Math.pow(Math.max(0, Math.log10(mEarth) + 1.2), 1.6) * rng.range(0.4, 1.5)));
    let ma = radiusAU * rng.range(2.2, 4.0);
    for (let m = 0; m < nMoons; m++) {
      if (ma > hill * 0.45) break;
      const mm = rng.powerLaw(0.0005, Math.min(0.05, mEarth * 0.02), -1.1);
      const mr = radiusFromMass(mm) / EARTH_RADII_PER_AU;
      planet.moons.push({
        semiMajorAU: ma,
        radiusAU: mr,
        massEarth: mm,
        eccentricity: rng.f() * 0.05,
        inclination: rng.normal() * 0.06,
        phase: rng.f() * Math.PI * 2,
        // Kepler again, now around the planet: period in years.
        periodYears: Math.sqrt(ma * ma * ma / (mEarth / 333000)),
        seed: rng.u32(),
        icy: planet.tEq < 260,
      });
      ma *= rng.range(1.5, 2.6);
    }

    planets.push(planet);

    // Next orbit, placed far enough out to be dynamically stable. Systems
    // packed tighter than about ten mutual Hill radii do not survive.
    const mu = mEarth / (333000 * star.mass);
    const rHill = a * Math.pow(mu / 3, 1 / 3);
    const spacing = rng.range(9, 26) * rHill;
    a = Math.max(a * rng.range(1.35, 2.2), a + spacing);
  }

  /* -- small bodies ---------------------------------------------------- */
  // An asteroid belt sits where a giant's resonances stirred the planetesimals
  // too violently for a planet to finish forming.
  const giants = planets.filter((p) => p.massEarth > 50);
  const belts = [];
  if (giants.length && rng.f() < 0.8) {
    const g = giants[0];
    const inner = g.semiMajorAU * rng.range(0.38, 0.55);
    belts.push({
      innerAU: inner, outerAU: inner * rng.range(1.25, 1.9),
      count: 2000 + rng.int(9000),
      inclinationSpread: rng.range(0.05, 0.22),
      seed: rng.u32(),
      kind: 'asteroid',
    });
  }
  // And a cold reservoir of ices beyond the outermost planet.
  if (planets.length) {
    const outer = planets[planets.length - 1].semiMajorAU;
    belts.push({
      innerAU: outer * rng.range(1.3, 1.9), outerAU: outer * rng.range(2.4, 4.5),
      count: 4000 + rng.int(12000),
      inclinationSpread: rng.range(0.15, 0.5),
      seed: rng.u32(),
      kind: 'kuiper',
    });
  }

  return {
    seed, star, companion, planets, belts,
    rInner, rSnow, rOuter, hzInner, hzOuter,
    habitableCount: planets.filter((p) => p.habitable).length,
    name: systemName(seed),
  };
}

/* ------------------------------------------------------------- mechanics -- */

// Solves Kepler's equation M = E - e sin E by Newton's method, with a starting
// guess good enough that three iterations converge to double precision for all
// but the most eccentric orbits.
export function eccentricAnomaly(M, e) {
  M = M % (Math.PI * 2);
  if (M < 0) M += Math.PI * 2;
  let E = e < 0.8 ? M : Math.PI;
  for (let i = 0; i < 12; i++) {
    const f = E - e * Math.sin(E) - M;
    const fp = 1 - e * Math.cos(E);
    const d = f / fp;
    E -= d;
    if (Math.abs(d) < 1e-12) break;
  }
  return E;
}

// Position of a body on its orbit at time t (years), in the system's frame.
export function orbitalPosition(el, tYears, out = [0, 0, 0]) {
  const n = (Math.PI * 2) / el.periodYears;
  const M = (el.meanAnomaly0 ?? el.phase ?? 0) + n * tYears;
  const e = el.eccentricity || 0;
  const E = eccentricAnomaly(M, e);
  const a = el.semiMajorAU;
  // Position in the orbital plane.
  const xv = a * (Math.cos(E) - e);
  const yv = a * Math.sqrt(1 - e * e) * Math.sin(E);

  const w = el.argumentPeriapsis || 0;
  const O = el.longitudeAscending || 0;
  const i = el.inclination || 0;
  const cw = Math.cos(w), sw = Math.sin(w);
  const cO = Math.cos(O), sO = Math.sin(O);
  const ci = Math.cos(i), si = Math.sin(i);

  // Rotate perifocal -> reference frame (z is the system's north pole).
  const x1 = xv * cw - yv * sw;
  const y1 = xv * sw + yv * cw;
  out[0] = x1 * cO - y1 * ci * sO;
  out[1] = y1 * si;
  out[2] = x1 * sO + y1 * ci * cO;
  return out;
}

// Orbital speed from the vis-viva equation, in AU per year.
export function orbitalSpeed(el, r, starMass) {
  return Math.sqrt(4 * Math.PI * Math.PI * starMass * (2 / r - 1 / el.semiMajorAU));
}

/* ------------------------------------------------------------------ names -- */

const SYL_A = ['ka', 've', 'thu', 'sol', 'ny', 'ar', 'zel', 'mir', 'ta', 'or', 'lu', 'xi', 'cor', 'bel', 'dra', 'ph'];
const SYL_B = ['ran', 'thos', 'mir', 'nex', 'dar', 'vess', 'lun', 'quor', 'nis', 'tal', 'zor', 'hem', 'ric', 'wen'];
const GREEK = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu'];

export function systemName(seed) {
  const rng = new RNG((seed >>> 0) ^ 0x5bf03635);
  const a = SYL_A[rng.int(SYL_A.length)];
  const b = SYL_B[rng.int(SYL_B.length)];
  const base = (a + b).replace(/^./, (c) => c.toUpperCase());
  const style = rng.int(3);
  if (style === 0) return `${GREEK[rng.int(GREEK.length)]} ${base}`;
  if (style === 1) return `${base}-${rng.int(900) + 100}`;
  return base;
}

export const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'];
