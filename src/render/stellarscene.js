// Draws a whole star system, and keeps track of where everything is.
//
// Bodies move on real Keplerian orbits advanced by a clock the user controls,
// so inner planets lap outer ones at the rate Kepler's third law demands, moons
// swing around their planets, and the shadows and phases follow from the actual
// geometry rather than from anything scripted.

import { orbitalPosition, PLANET_TYPE, ROMAN } from '../universe/system.js';
import { v3, v3set, v3sub, v3len, v3norm, v3copy } from '../core/math.js';

// How much atmosphere each kind of world carries, as a fraction of its radius.
const ATMOSPHERE = {
  [PLANET_TYPE.LAVA]: 0.016,
  [PLANET_TYPE.ROCK]: 0.0,
  [PLANET_TYPE.DESERT]: 0.010,
  [PLANET_TYPE.TERRAN]: 0.026,
  [PLANET_TYPE.OCEAN]: 0.030,
  [PLANET_TYPE.ICE]: 0.005,
  [PLANET_TYPE.ICE_GIANT]: 0.055,
  [PLANET_TYPE.GAS_GIANT]: 0.050,
};
const CLOUDINESS = {
  [PLANET_TYPE.LAVA]: 0.25,
  [PLANET_TYPE.ROCK]: 0.0,
  [PLANET_TYPE.DESERT]: 0.12,
  [PLANET_TYPE.TERRAN]: 0.45,
  [PLANET_TYPE.OCEAN]: 0.58,
  [PLANET_TYPE.ICE]: 0.18,
  [PLANET_TYPE.ICE_GIANT]: 0.0,
  [PLANET_TYPE.GAS_GIANT]: 0.0,
};

export class StellarScene {
  constructor(bodyRenderer, starfield, nebulae = null) {
    this.bodies = bodyRenderer;
    this.starfield = starfield;
    this.nebulae = nebulae;
    this.system = null;
    this.timeYears = 0;
    this.timeRate = 0.08;          // years of simulated time per second
    this.exposure = 1;
    this.showOrbits = true;
    // Every body's current world position, refreshed each frame, so the camera,
    // targeting and HUD all read the same state the renderer draws.
    this.placed = [];
  }

  setSystem(system) {
    this.system = system;
    this.timeYears = 0;
    this.placed = [];
  }

  advance(dt) { this.timeYears += dt * this.timeRate; }

  // Recomputes every body's position for the current time.
  layout() {
    const s = this.system;
    if (!s) return this.placed;
    const out = [];
    out.push({
      kind: 'star', name: s.name, ref: s.star,
      pos: v3(0, 0, 0), radius: s.star.radiusAU,
      label: `${s.name} · ${s.star.class}-type star`,
    });

    if (s.companion) {
      const c = s.companion;
      const ang = c.phase + this.timeYears * (2 * Math.PI) /
        Math.sqrt(Math.pow(c.separationAU, 3) / (s.star.mass + c.mass));
      out.push({
        kind: 'star', name: `${s.name} B`, ref: c,
        pos: v3(Math.cos(ang) * c.separationAU, 0, Math.sin(ang) * c.separationAU),
        radius: c.radiusAU,
        label: `${s.name} B · ${c.class}-type companion`,
      });
    }

    for (const p of s.planets) {
      const pos = orbitalPosition(p, this.timeYears, [0, 0, 0]);
      const entry = {
        kind: 'planet', ref: p,
        pos: v3(pos[0], pos[1], pos[2]),
        radius: p.radiusAU,
        label: `${s.name} ${ROMAN[p.index] || p.index + 1} · ${p.typeName}`,
      };
      out.push(entry);
      for (let mi = 0; mi < p.moons.length; mi++) {
        const m = p.moons[mi];
        const mp = orbitalPosition({ ...m, meanAnomaly0: m.phase, argumentPeriapsis: 0, longitudeAscending: 0 },
          this.timeYears, [0, 0, 0]);
        out.push({
          kind: 'moon', ref: m, parent: entry,
          pos: v3(pos[0] + mp[0], pos[1] + mp[1], pos[2] + mp[2]),
          radius: m.radiusAU,
          label: `${s.name} ${ROMAN[p.index] || p.index + 1}${String.fromCharCode(97 + mi)} · moon`,
        });
      }
    }
    this.placed = out;
    return out;
  }

  // Distance from the camera to the nearest body surface, for the near plane.
  nearestSurface(cameraPos) {
    let best = Infinity;
    for (const b of this.placed) {
      const d = Math.hypot(b.pos[0] - cameraPos[0], b.pos[1] - cameraPos[1], b.pos[2] - cameraPos[2]) - b.radius;
      if (d < best) best = d;
    }
    return best;
  }

  render(camera, { time = 0 } = {}) {
    const s = this.system;
    if (!s) return;
    const star = s.star;

    // Sky first: stars, then nebulae over them, so a dark nebula genuinely
    // obscures the star field behind it rather than merely adding nothing.
    this.starfield.render(camera);
    if (this.nebulae) this.nebulae.render(camera, time);

    const bodies = this.bodies;
    bodies.begin(camera);

    const cam = camera.position;
    const rel = (p) => v3(p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]);

    // Painter's order: the atmospheres blend, so far bodies must be laid down
    // before near ones.
    const sorted = [...this.placed].sort((a, b) => {
      const da = (a.pos[0] - cam[0]) ** 2 + (a.pos[1] - cam[1]) ** 2 + (a.pos[2] - cam[2]) ** 2;
      const db = (b.pos[0] - cam[0]) ** 2 + (b.pos[1] - cam[1]) ** 2 + (b.pos[2] - cam[2]) ** 2;
      return db - da;
    });

    // Orbit paths, faint, behind everything.
    if (this.showOrbits) {
      const originRel = rel([0, 0, 0]);
      for (const p of s.planets) {
        // Fade out once the camera is close enough that a line across the sky
        // would be a distraction rather than a map.
        const camR = Math.hypot(cam[0], cam[1], cam[2]);
        const opacity = 0.05 * (1 - Math.exp(-camR / (p.semiMajorAU * 0.35)));
        if (opacity < 0.002) continue;
        const warm = p.habitable ? [0.55, 0.85, 1.0] : [0.55, 0.62, 0.85];
        bodies.drawOrbit(camera, {
          centre: originRel,
          semiMajorAU: p.semiMajorAU,
          eccentricity: p.eccentricity,
          inclination: p.inclination,
          longitudeAscending: p.longitudeAscending,
          argumentPeriapsis: p.argumentPeriapsis,
          colour: warm,
          opacity,
          segments: 192,
        });
      }
    }

    // Debris rings next: they sit behind everything and only add light.
    for (const belt of s.belts) {
      bodies.drawBelt(camera, {
        centre: rel([0, 0, 0]),
        inner: belt.innerAU, outer: belt.outerAU,
        incSpread: belt.inclinationSpread,
        seed: belt.seed, time: this.timeYears,
        starMass: star.mass,
        size: belt.kind === 'kuiper' ? 4e-6 : 3e-6,
        brightness: (belt.kind === 'kuiper' ? 0.06 : 0.16) * star.luminosity,
        sunColour: star.color,
        count: belt.count,
      });
    }

    for (const b of sorted) {
      if (b.kind === 'star') {
        bodies.drawStar(camera, {
          centre: rel(b.pos),
          radius: b.radius,
          temperature: b.ref.temperature,
          luminosity: b.ref.luminosity,
          // Surface brightness of a star does not depend on distance; the glare
          // around it does, and both are handled in the shader.
          surface: 9.0 * this.exposure,
          glare: 26.0 * this.exposure,
          time, seed: (b.ref.mass * 1000) % 1000,
        });
        continue;
      }

      const p = b.ref;
      const isMoon = b.kind === 'moon';
      // Irradiance at this body, in units of the flux at 1 AU from a solar star.
      const d = Math.max(v3len(b.pos), 1e-6);
      const irradiance = star.luminosity / (d * d);
      const sunDir = v3norm(v3(), v3(-b.pos[0], -b.pos[1], -b.pos[2]));
      const sunColour = [
        star.color[0] * irradiance, star.color[1] * irradiance, star.color[2] * irradiance,
      ];

      const type = isMoon ? (p.icy ? PLANET_TYPE.ICE : PLANET_TYPE.ROCK) : p.type;
      const spin = p.tidallyLocked
        ? Math.atan2(b.pos[2], b.pos[0])
        : (this.timeYears * 8766 / (p.rotationHours || 24)) % (Math.PI * 2);

      // Obliquity tilts the spin axis away from the orbital pole, which is what
      // gives a world seasons and an asymmetric terminator.
      const ob = isMoon ? 0.05 : p.obliquity;
      const axis = v3(Math.sin(ob), Math.cos(ob), 0);

      bodies.drawPlanet(camera, {
        centre: rel(b.pos),
        radius: b.radius,
        atmoScale: isMoon ? 0 : (ATMOSPHERE[type] ?? 0),
        sunDir, sunColour,
        axis, spin,
        seed: (p.seed % 4096) / 7.13,
        type,
        temperature: p.tSurface || p.tEq || 200,
        cloudiness: isMoon ? 0 : (CLOUDINESS[type] ?? 0),
        oceanLevel: type === PLANET_TYPE.OCEAN ? 0.62 : 0.48,
        iceLatitude: Math.max(0.35, Math.min(0.95, 1 - (p.tSurface || 280) / 420)),
        ringInner: isMoon ? 0 : p.ringInner,
        ringOuter: isMoon ? 0 : p.ringOuter,
        ringOpacity: isMoon ? 0 : p.ringOpacity,
        ringAxis: axis,
        ringSeed: (p.seed % 997) / 13.7,
        habitable: !isMoon && p.habitable,
        exposure: this.exposure,
        time,
      });
    }

    bodies.end();
  }
}
