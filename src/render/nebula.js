// Nebulae, rendered as volumes on the sky.
//
// A star system is a few hundred astronomical units across; the nearest nebula
// is hundreds of light years away. Flying around the system produces no
// measurable parallax against it, so a nebula is correctly drawn as a fixed
// feature of the sky - but it is still marched as a real three-dimensional
// volume, so it has depth, self-shadowing dust, and an interior.
//
// Colour is spectroscopic rather than decorative. An HII region glows in the
// hydrogen Balmer lines - deep red H-alpha at 656 nm with blue H-beta at 486 nm,
// which together read magenta, not the red of a narrowband image. Where the
// ionising star is hot enough to strip oxygen twice, the forbidden [O III]
// doublet at 500 nm adds teal to the core. Reflection nebulae are blue because
// dust grains scatter short wavelengths more efficiently, for the same reason
// the sky is.

import { HASH_GLSL, BLACKBODY_GLSL } from './color.js';
import { RNG } from '../core/rng.js';
import { m4, m4mul } from '../core/math.js';

export const NEBULA_TYPE = { EMISSION: 0, REFLECTION: 1, DARK: 2, REMNANT: 3, PLANETARY: 4 };

const VS = `#version 300 es
precision highp float;
uniform mat4 uViewProjRel;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uDir;         // direction to the nebula
uniform float uAngular;    // angular radius, radians
out vec3 vRay;

void main() {
  vec2 quad[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                         vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  vec2 c = quad[gl_VertexID];
  float s = tan(min(uAngular, 1.2));
  vec3 p = normalize(uDir) + uCamRight * (c.x * s) + uCamUp * (c.y * s);
  vRay = p;
  gl_Position = uViewProjRel * vec4(p * 1.0e7, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vRay;
out vec4 fragColor;

uniform vec3 uDir;
uniform float uAngular;
uniform float uSeed;
uniform int uType;
uniform float uBrightness;
uniform float uIonisation;   // how hard the illuminating stars are
uniform float uDustiness;
uniform vec3 uTint;
uniform float uTime;
${HASH_GLSL}
${BLACKBODY_GLSL}

// Turbulent filamentary structure. Real nebulae are shaped by shocks and
// ionisation fronts, so a plain fractal is too smooth - warping the domain by
// another fractal produces the sheets and pillars.
float nebulaDensity(vec3 p, out float ionised) {
  vec3 warp = vec3(fbm(p * 0.9 + 11.0, 4), fbm(p * 0.9 + 27.0, 4), fbm(p * 0.9 + 41.0, 4));
  vec3 q = p + warp * 1.35;
  float base = fbm(q * 1.15, 5);
  float filament = 1.0 - abs(fbm(q * 2.1 + 5.0, 4));      // ridges
  float d = base * 0.55 + filament * 0.55;
  // Confine it to a rough ellipsoid so it has an edge.
  float r = length(p);
  d *= 1.0 - smoothstep(0.35, 1.0, r);
  // The interior is more highly ionised: that is where the hot stars are.
  ionised = clamp((1.0 - r * 1.4) * uIonisation + fbm(q * 3.0, 3) * 0.35, 0.0, 1.0);
  return max(d - 0.16, 0.0);
}

void main() {
  vec3 rd = normalize(vRay);
  vec3 axis = normalize(uDir);
  // Angular offset from the centre, mapped into the volume's own coordinates.
  vec3 t = abs(axis.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : normalize(cross(vec3(0.0, 1.0, 0.0), axis));
  vec3 b = cross(axis, t);
  float s = tan(min(uAngular, 1.2));
  vec2 uv = vec2(dot(rd, t), dot(rd, b)) / max(s * max(dot(rd, axis), 1e-3), 1e-6);
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;

  // March along the depth axis of a unit sphere at the sampled offset.
  float depth = sqrt(max(1.0 - r2, 0.0));
  const int STEPS = 22;
  float dt = 2.0 * depth / float(STEPS);
  float jitter = hash1(uint(gl_FragCoord.x) * 7919u + uint(gl_FragCoord.y) * 104729u + uint(uSeed));

  vec3 emission = vec3(0.0);
  float trans = 1.0;
  float seedOff = uSeed * 0.017;

  // Line colours, in linear sRGB. H-alpha and H-beta together are magenta;
  // [O III] is the teal that shows up in the hottest cores.
  const vec3 HALPHA = vec3(1.00, 0.09, 0.13);
  const vec3 HBETA  = vec3(0.26, 0.42, 1.00);
  const vec3 OIII   = vec3(0.13, 0.95, 0.62);

  for (int i = 0; i < STEPS; i++) {
    float z = -depth + dt * (float(i) + jitter);
    vec3 p = vec3(uv, z);
    p += vec3(seedOff, seedOff * 1.7, seedOff * 0.3);
    float ion;
    float d = nebulaDensity(p, ion);
    if (d <= 0.0) continue;

    vec3 c;
    if (uType == 0) {                       // emission / HII region
      c = HALPHA * 1.0 + HBETA * 0.34 + OIII * (ion * ion * 1.3);
    } else if (uType == 1) {                // reflection: scattered starlight
      c = vec3(0.30, 0.48, 1.00);
    } else if (uType == 3) {                // supernova remnant: shocked shell
      float shell = smoothstep(0.55, 0.85, length(p)) * (1.0 - smoothstep(0.85, 1.0, length(p)));
      c = (OIII * 0.9 + HALPHA * 0.7) * (0.35 + 2.4 * shell);
    } else if (uType == 4) {                // planetary nebula: ionised shell
      float shell = smoothstep(0.42, 0.62, length(p)) * (1.0 - smoothstep(0.72, 0.95, length(p)));
      c = (OIII * 1.5 + HBETA * 0.6 + HALPHA * 0.5) * (0.1 + 3.2 * shell);
    } else {                                // dark nebula: absorption only
      c = vec3(0.0);
    }
    c *= uTint;

    float dust = uDustiness * d;
    emission += trans * c * d * dt;
    trans *= exp(-dust * dt * 5.0);
    if (trans < 0.02) break;
  }

  vec3 col = emission * uBrightness;
  float alpha = clamp(1.0 - trans, 0.0, 1.0);
  if (uType == 2) col = vec3(0.0);          // pure obscuration

  // Soft edge so the bounding disc never shows.
  float edge = 1.0 - smoothstep(0.75, 1.0, sqrt(r2));
  col *= edge; alpha *= edge;
  if (alpha < 0.002 && dot(col, vec3(1.0)) < 1e-4) discard;

  fragColor = vec4(col, alpha);
}`;

// Builds a plausible set of nebulae for a galaxy, seen from a point inside it.
// Star-forming galaxies are full of HII regions; ellipticals have almost none,
// because they have almost no cold gas left.
export function generateNebulae(seed, { young = 0.5, elliptical = false, count = null } = {}) {
  const rng = new RNG(seed >>> 0);
  const n = count ?? (elliptical ? rng.int(2) : 3 + rng.int(7) + Math.round(young * 5));
  const out = [];
  for (let i = 0; i < n; i++) {
    const d = rng.onSphere([0, 0, 0]);
    // Nebulae live in the disk, so bias directions toward the galactic plane.
    d[1] *= 0.22;
    const len = Math.hypot(d[0], d[1], d[2]) || 1;
    const dir = [d[0] / len, d[1] / len, d[2] / len];

    let type;
    const u = rng.f();
    if (elliptical) type = u < 0.6 ? NEBULA_TYPE.PLANETARY : NEBULA_TYPE.REMNANT;
    else if (u < 0.48) type = NEBULA_TYPE.EMISSION;
    else if (u < 0.68) type = NEBULA_TYPE.REFLECTION;
    else if (u < 0.82) type = NEBULA_TYPE.DARK;
    else if (u < 0.93) type = NEBULA_TYPE.REMNANT;
    else type = NEBULA_TYPE.PLANETARY;

    // Angular size: a nearby star-forming complex can span tens of degrees,
    // a planetary nebula only arcminutes.
    const base = type === NEBULA_TYPE.PLANETARY ? 0.012 : type === NEBULA_TYPE.REMNANT ? 0.09 : 0.16;
    out.push({
      dir,
      angular: base * rng.range(0.45, 2.6),
      type,
      seed: rng.u32() % 4096,
      ionisation: type === NEBULA_TYPE.EMISSION ? rng.range(0.35, 1.0) : rng.range(0.1, 0.5),
      dustiness: type === NEBULA_TYPE.DARK ? rng.range(1.4, 2.6) : rng.range(0.25, 0.9),
      brightness: type === NEBULA_TYPE.DARK ? 0 : rng.range(0.35, 1.4),
      tint: [rng.range(0.85, 1.15), rng.range(0.85, 1.1), rng.range(0.9, 1.2)],
    });
  }
  // Draw the dark ones last so they obscure what is behind them.
  out.sort((a, b) => (a.type === NEBULA_TYPE.DARK ? 1 : 0) - (b.type === NEBULA_TYPE.DARK ? 1 : 0));
  return out;
}

export class NebulaRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.prog = ctx.program(VS, FS, 'nebula');
    this.vao = ctx.gl.createVertexArray();
    this.nebulae = [];
    this.brightness = 0.5;
    this._vp = m4();
  }

  set(nebulae) { this.nebulae = nebulae || []; }

  render(camera, time = 0) {
    if (!this.nebulae.length) return;
    const gl = this.gl;
    const viewRel = m4();
    camera.viewNoTranslation(viewRel);
    m4mul(this._vp, camera.proj, viewRel);

    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    // Premultiplied source-over, so dark nebulae genuinely block the stars
    // behind them instead of merely adding nothing.
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.vao);

    const p = this.prog.use()
      .set('uViewProjRel', this._vp)
      .set('uCamRight', camera.right)
      .set('uCamUp', camera.up)
      .set('uTime', time);

    for (const n of this.nebulae) {
      p.set('uDir', new Float32Array(n.dir))
        .set('uAngular', n.angular)
        .set('uSeed', n.seed)
        .set('uType', n.type)
        .set('uIonisation', n.ionisation)
        .set('uDustiness', n.dustiness)
        .set('uBrightness', n.brightness * this.brightness)
        .set('uTint', new Float32Array(n.tint));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      this.ctx.drawCalls++;
    }

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }
}
