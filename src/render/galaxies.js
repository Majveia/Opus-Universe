// Draws galaxies.
//
// Far away they are points. As one fills more of the screen it is raymarched as
// a small volume: an exponential disk with a vertical scale height, logarithmic
// spiral arms, a Sersic bulge, and an absorbing dust layer offset to the
// leading edge of the arms. Marching a volume rather than compositing a flat
// sprite is what makes inclination work - a disk seen edge-on shows a dust lane
// cutting across a bright bulge, and the same code produces it from the same
// parameters that produce the face-on spiral.

import { BLACKBODY_GLSL, HASH_GLSL } from './color.js';
import { GALAXY_TEXELS } from '../universe/galaxies.js';
import { v3, v3set, m4, m4mul, sphereInFrustum } from '../core/math.js';

const GALAXY_COMMON = `
uniform sampler2D uGalaxies;
uniform int uCount;

struct Galaxy {
  vec3 pos; float radius;
  vec3 axis; float type;
  vec3 u; float axisRatio;
  vec3 v; float bulgeFraction;
  vec3 color; float logL;
  float armCount, pitch, armStrength, dust;
  float bar, young, seed, logMstar;
};

Galaxy readGalaxy(int i) {
  Galaxy g;
  vec4 a = texelFetch(uGalaxies, ivec2(0, i), 0);
  vec4 b = texelFetch(uGalaxies, ivec2(1, i), 0);
  vec4 c = texelFetch(uGalaxies, ivec2(2, i), 0);
  vec4 d = texelFetch(uGalaxies, ivec2(3, i), 0);
  vec4 e = texelFetch(uGalaxies, ivec2(4, i), 0);
  vec4 f = texelFetch(uGalaxies, ivec2(5, i), 0);
  vec4 h = texelFetch(uGalaxies, ivec2(6, i), 0);
  g.pos = a.xyz;   g.radius = a.w;
  g.axis = b.xyz;  g.type = b.w;
  g.u = c.xyz;     g.axisRatio = c.w;
  g.v = d.xyz;     g.bulgeFraction = d.w;
  g.color = e.xyz; g.logL = e.w;
  g.armCount = f.x; g.pitch = f.y; g.armStrength = f.z; g.dust = f.w;
  g.bar = h.x; g.young = h.y; g.seed = h.z; g.logMstar = h.w;
  return g;
}`;

/* ------------------------------------------------------------ point pass -- */

const POINT_VS = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform mat4 uViewProjRel;
uniform vec3 uCameraPos;
uniform vec3 uTileOffset;
uniform float uPixelsPerRadian;
uniform float uBrightness;
uniform float uRefDistance;
${GALAXY_COMMON}
out vec3 vColor;
out float vIntensity;

void main() {
  Galaxy g = readGalaxy(gl_VertexID);
  vec3 rel = g.pos + uTileOffset - uCameraPos;
  float dist = length(rel);
  gl_Position = uViewProjRel * vec4(rel, 1.0);

  float angPx = g.radius * uPixelsPerRadian / max(dist, 1e-6);
  gl_PointSize = clamp(angPx * 1.6, 1.0, 20.0);

  // Inverse-square flux, normalised so a 10^10 Lsun galaxy one box away sits
  // near unit brightness. Hand over to the volumetric pass once resolved.
  float d = dist / uRefDistance;
  float flux = pow(10.0, g.logL - 10.0) / max(d * d, 1e-8);
  float handover = 1.0 - smoothstep(1.5, 5.0, angPx);
  vIntensity = uBrightness * flux * handover;
  vColor = g.color;
  if (vIntensity < 1e-5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;

const POINT_FS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vIntensity;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float g = exp(-r2 * 3.2) + 0.10 * exp(-r2 * 0.8);
  fragColor = vec4(vColor * (vIntensity * g), 1.0);
}`;

/* ------------------------------------------------------ volumetric pass -- */

const VOLUME_VS = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform mat4 uViewProjRel;
uniform vec3 uCameraPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uTileOffset;
uniform float uPixelsPerRadian;
uniform float uExtent;          // billboard half-size in disk scale radii
${GALAXY_COMMON}

flat out int vIndex;
out vec3 vRayDir;
flat out vec3 vCentre;
flat out float vAngPx;

void main() {
  int id = gl_InstanceID;
  Galaxy g = readGalaxy(id);
  vec3 rel = g.pos + uTileOffset - uCameraPos;
  float dist = length(rel);
  float angPx = g.radius * uPixelsPerRadian / max(dist, 1e-6);

  // Collapse the quad for anything the point pass is still handling.
  if (angPx < 1.2) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vIndex = -1; vRayDir = vec3(0.0); vCentre = vec3(0.0); vAngPx = 0.0; return; }

  // Two triangles making a camera-facing quad.
  vec2 quad[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                         vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  vec2 corner = quad[gl_VertexID];

  float halfSize = g.radius * uExtent;
  vec3 offset = uCamRight * (corner.x * halfSize) + uCamUp * (corner.y * halfSize);
  vec3 p = rel + offset;

  gl_Position = uViewProjRel * vec4(p, 1.0);
  vIndex = id;
  vRayDir = p;
  vCentre = rel;
  vAngPx = angPx;
}`;

const VOLUME_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
flat in int vIndex;
in vec3 vRayDir;
flat in vec3 vCentre;
flat in float vAngPx;
out vec4 fragColor;

uniform float uExtent;
uniform float uBrightness;
uniform float uRefDistance;
uniform float uTime;
${GALAXY_COMMON}
${BLACKBODY_GLSL}
${HASH_GLSL}

// Density and emission of the galaxy at a point in its own frame, with the
// disk in the xy plane and lengths in units of the disk scale radius.
// Returns emission in rgb and dust opacity in a.
vec4 galaxyMedium(Galaxy g, vec3 q, float seed) {
  float r = length(q.xy);
  float z = q.z;

  // --- spheroid -------------------------------------------------------
  // de Vaucouleurs profile, flattened by the measured axis ratio.
  vec3 sq = vec3(q.xy, q.z / max(g.axisRatio, 0.2));
  float rs = max(length(sq), 0.02);
  float rBulge = (g.type < 0.5) ? 1.0 : 0.16;
  float bulge = exp(-7.669 * (pow(rs / rBulge, 0.25) - 1.0));
  bulge = min(bulge, 60.0);

  if (g.type < 0.5) {
    // Elliptical: spheroid only, an old red population, faint dust.
    vec3 cOld = blackbodyRGB(3900.0 + 500.0 * g.young);
    return vec4(cOld * bulge * 0.55, bulge * 0.004);
  }

  // --- disk ------------------------------------------------------------
  float hz = 0.10 + 0.05 * g.young;               // vertical scale height
  float vertical = exp(-abs(z) / hz);
  float radial = exp(-r);

  // Logarithmic spiral arms. The phase winds as ln(r)/tan(pitch), which is what
  // makes a spiral self-similar under scaling.
  float phi = atan(q.y, q.x);
  float wind = log(max(r, 0.06)) / max(tan(g.pitch), 0.05);
  float armPhase = g.armCount * (phi - wind);

  // A central bar rotates the inner isophotes and feeds the arms.
  float barMask = g.bar * exp(-pow(r / 0.45, 2.0));
  armPhase += barMask * 2.0;

  float arm = pow(0.5 + 0.5 * cos(armPhase), 3.0);
  // Arms fade out in the very centre and in the far outskirts.
  arm *= smoothstep(0.05, 0.35, r) * (1.0 - smoothstep(2.4, 3.6, r));

  float disk = radial * vertical * (1.0 + g.armStrength * 2.6 * arm);

  // Flocculent structure: real arms are not clean sinusoids.
  float floc = 0.5 + 0.5 * fbm(vec3(q.xy * 3.2, seed * 0.01), 4);
  disk *= 0.55 + 0.9 * floc;

  // --- populations ------------------------------------------------------
  // Young blue stars form in the arms, where the density wave compresses gas.
  vec3 cOld = blackbodyRGB(4200.0);
  vec3 cYoung = blackbodyRGB(14000.0);
  float youngFrac = clamp(g.young * (0.25 + 1.5 * arm), 0.0, 1.0);
  vec3 emission = mix(cOld, cYoung, youngFrac) * disk;

  // HII regions: bright, discrete, and always sitting on the arms.
  float knot = fbm(vec3(q.xy * 14.0, seed * 0.03), 3);
  float hii = smoothstep(0.42, 0.72, knot) * arm * g.young;
  emission += blackbodyRGB(9000.0) * hii * 3.0 * radial * vertical;

  emission += cOld * bulge * g.bulgeFraction * 0.5;

  // --- dust -------------------------------------------------------------
  // Dust sits in a thinner layer than the stars and slightly inside the arms,
  // which is why spiral dust lanes appear on the concave edge.
  float dustPhase = pow(0.5 + 0.5 * cos(armPhase + 0.55), 3.0);
  float dustVert = exp(-abs(z) / (hz * 0.45));
  float dust = g.dust * radial * dustVert * (0.35 + 2.0 * dustPhase)
             * (0.6 + 0.8 * fbm(vec3(q.xy * 5.0, seed * 0.02 + 11.0), 4));

  return vec4(emission, max(dust, 0.0));
}

void main() {
  if (vIndex < 0) discard;
  Galaxy g = readGalaxy(vIndex);

  vec3 rd = normalize(vRayDir);
  vec3 oc = -vCentre;                       // camera-to-centre, camera at origin
  float R = g.radius * uExtent;

  // Ray-sphere intersection bounding the galaxy.
  float b = dot(rd, oc);
  float c = dot(oc, oc) - R * R;
  float disc = b * b - c;
  if (disc <= 0.0) discard;
  float sq = sqrt(disc);
  float t0 = max(-b - sq, 0.0);
  float t1 = -b + sq;
  if (t1 <= t0) discard;

  // More screen area earns more samples; a distant galaxy needs very few.
  int steps = int(clamp(vAngPx * 0.55, 6.0, 40.0));
  float dt = (t1 - t0) / float(steps);

  // Jittering the entry point turns banding into noise, which the bloom and
  // the film grain then hide completely.
  float jitter = hash1(uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + uint(vIndex) * 26699u);
  float t = t0 + dt * jitter;

  vec3 accum = vec3(0.0);
  float trans = 1.0;
  float invR = 1.0 / max(g.radius, 1e-9);
  float seed = g.seed;

  for (int i = 0; i < 40; i++) {
    if (i >= steps || trans < 0.01) break;
    vec3 rel = rd * t - vCentre;              // sample point, relative to the galaxy centre
    // Into the galaxy's own frame, in units of the disk scale radius.
    vec3 q = vec3(dot(rel, g.u), dot(rel, g.v), dot(rel, g.axis)) * invR;

    vec4 m = galaxyMedium(g, q, seed);
    float extinction = m.a * dt * invR;
    accum += trans * m.rgb * dt * invR;
    trans *= exp(-extinction * 6.0);
    t += dt;
  }

  float dist = length(vCentre) / uRefDistance;
  // The march already integrates surface brightness, so only the inverse-square
  // dimming of the whole system remains.
  float scale = uBrightness * pow(10.0, g.logL - 10.0) / max(dist * dist, 1e-8);
  vec3 col = accum * scale * 0.06;

  fragColor = vec4(col, 1.0);
}`;

export class GalaxyRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.progPoint = ctx.program(POINT_VS, POINT_FS, 'galaxy-points');
    this.progVolume = ctx.program(VOLUME_VS, VOLUME_FS, 'galaxy-volume');
    this.vao = ctx.gl.createVertexArray();
    this.texture = null;
    this.count = 0;
    this.settings = {
      brightness: 0.055,
      extent: 3.4,
      tileRadius: 1,
      volumetric: true,
    };
    this._tmpM = m4();
    this._tile = v3();
    this._stats = { drawn: 0, tiles: 0 };
  }

  get stats() { return this._stats; }

  // Uploads a packed population. One row per galaxy keeps indexing trivial.
  setGalaxies(packed, count, population = null) {
    this.lastPopulation = population;
    const gl = this.gl;
    if (this.texture) this.texture.dispose();
    this.count = count;
    if (!count) { this.texture = null; return; }
    const maxRows = this.ctx.limits.maxTextureSize;
    if (count > maxRows) count = this.count = maxRows;
    this.texture = this.ctx.texture({
      width: GALAXY_TEXELS, height: count, format: 'RGBA32F',
      filter: gl.NEAREST, wrap: gl.CLAMP_TO_EDGE,
    });
    this.texture.upload(packed.subarray(0, GALAXY_TEXELS * count * 4));
  }

  render(camera, boxWorldSize) {
    if (!this.texture || !this.count) return;
    const gl = this.gl;
    const s = this.settings;

    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);

    const viewRel = m4();
    camera.viewNoTranslation(viewRel);
    m4mul(this._tmpM, camera.proj, viewRel);
    const pixelsPerRadian = (gl.drawingBufferHeight * 0.5) / Math.tan(camera.fov * 0.5);

    const tiles = [];
    const R = s.tileRadius;
    const cx = Math.floor(camera.position[0] / boxWorldSize);
    const cy = Math.floor(camera.position[1] / boxWorldSize);
    const cz = Math.floor(camera.position[2] / boxWorldSize);
    const cull = boxWorldSize * 0.8661;
    for (let iz = -R; iz <= R; iz++) for (let iy = -R; iy <= R; iy++) for (let ix = -R; ix <= R; ix++) {
      const ox = (cx + ix) * boxWorldSize, oy = (cy + iy) * boxWorldSize, oz = (cz + iz) * boxWorldSize;
      const mx = ox + boxWorldSize * 0.5, my = oy + boxWorldSize * 0.5, mz = oz + boxWorldSize * 0.5;
      if (!sphereInFrustum(camera.frustum, mx, my, mz, cull)) continue;
      tiles.push([ox, oy, oz]);
    }

    gl.bindVertexArray(this.vao);
    this._stats.tiles = tiles.length;
    this._stats.drawn = 0;

    /* -- unresolved galaxies as points -------------------------------- */
    const p = this.progPoint.use()
      .set('uViewProjRel', this._tmpM)
      .set('uCameraPos', camera.position)
      .set('uPixelsPerRadian', pixelsPerRadian)
      .set('uBrightness', s.brightness)
      .set('uRefDistance', boxWorldSize)
      .set('uCount', this.count);
    p.tex('uGalaxies', this.texture);
    for (const t of tiles) {
      v3set(this._tile, t[0], t[1], t[2]);
      p.set('uTileOffset', this._tile);
      gl.drawArrays(gl.POINTS, 0, this.count);
      this._stats.drawn += this.count;
      this.ctx.drawCalls++;
    }

    /* -- resolved galaxies raymarched --------------------------------- */
    if (s.volumetric) {
      const v = this.progVolume.use()
        .set('uViewProjRel', this._tmpM)
        .set('uCameraPos', camera.position)
        .set('uCamRight', camera.right)
        .set('uCamUp', camera.up)
        .set('uPixelsPerRadian', pixelsPerRadian)
        .set('uExtent', s.extent)
        .set('uBrightness', s.brightness)
        .set('uRefDistance', boxWorldSize)
        .set('uCount', this.count);
      v.tex('uGalaxies', this.texture);
      for (const t of tiles) {
        v3set(this._tile, t[0], t[1], t[2]);
        v.set('uTileOffset', this._tile);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.count);
        this.ctx.drawCalls++;
      }
    }

    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }
}
