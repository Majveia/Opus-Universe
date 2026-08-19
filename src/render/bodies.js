// Stars, planets, moons, rings and small bodies.
//
// Every body is drawn as a camera-facing quad whose fragment shader intersects
// the ray with an analytic sphere. There is no mesh and no level of detail: a
// planet is perfectly round from orbit and from a thousand astronomical units,
// and the silhouette never faceted.
//
// Atmospheres are integrated rather than faked. Rayleigh and Mie scattering
// coefficients, scale heights and the phase functions are the physical ones, so
// a thick atmosphere reddens toward its terminator, the limb glows, and the
// shadowed side keeps a thin bright rim - all of it falling out of the same
// integral rather than out of separate artistic passes.

import { BLACKBODY_GLSL, HASH_GLSL } from './color.js';
import { v3, m4, m4mul } from '../core/math.js';

const IMPOSTOR_VS = `#version 300 es
precision highp float;
uniform mat4 uViewProjRel;
uniform vec3 uCentre;        // body centre, relative to the camera
uniform float uQuadRadius;   // half-size of the billboard, world units
uniform vec3 uCamRight;
uniform vec3 uCamUp;
out vec3 vRay;

void main() {
  vec2 quad[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                         vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  vec2 c = quad[gl_VertexID];
  // The quad is pushed toward the camera by its own radius so a body the
  // camera is inside of still covers the screen.
  vec3 p = uCentre + uCamRight * (c.x * uQuadRadius) + uCamUp * (c.y * uQuadRadius);
  vRay = p;
  gl_Position = uViewProjRel * vec4(p, 1.0);
}`;

/* --------------------------------------------------------------- shared -- */

const SPHERE_GLSL = `
// Nearest intersection of a ray from the origin with a sphere at c of radius r.
// Returns vec2(t_near, t_far), or vec2(-1) on a miss.
vec2 raySphere(vec3 rd, vec3 c, float r) {
  float b = dot(rd, -c);
  float k = dot(c, c) - r * r;
  float disc = b * b - k;
  if (disc < 0.0) return vec2(-1.0);
  float s = sqrt(disc);
  return vec2(-b - s, -b + s);
}

// Rotates a direction into the body's own frame: spin about its axis, with the
// axis itself tilted away from the orbital pole by the obliquity.
mat3 bodyFrame(vec3 axis, float spin) {
  vec3 n = normalize(axis);
  vec3 t = abs(n.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : normalize(cross(vec3(0.0, 1.0, 0.0), n));
  vec3 b = cross(n, t);
  float cs = cos(spin), sn = sin(spin);
  vec3 t2 = t * cs + b * sn;
  vec3 b2 = -t * sn + b * cs;
  return mat3(t2, b2, n);           // columns; multiply as frame * v
}
`;

const ATMOSPHERE_GLSL = `
// Single-scattering atmosphere. Optical depth is accumulated along the view ray
// and again along the ray to the star from each sample, which is what produces
// reddening at grazing angles: blue light is scattered out of a long path
// before it can be scattered toward the eye.
struct Atmo {
  float planetR;
  float atmoR;
  float hRayleigh;      // scale height, in planet radii
  float hMie;
  vec3 betaR;           // scattering coefficients, per planet radius
  float betaM;
  float g;              // Mie asymmetry
};

float atmoDensityR(Atmo a, float h) { return exp(-h / a.hRayleigh); }
float atmoDensityM(Atmo a, float h) { return exp(-h / a.hMie); }

// Optical depth from p toward dir until leaving the atmosphere.
vec2 opticalDepthLight(Atmo a, vec3 p, vec3 dir, int steps) {
  vec2 t = raySphere(dir, -p, a.atmoR);      // from p, in local coordinates
  if (t.y <= 0.0) return vec2(0.0);
  float len = t.y / float(steps);
  vec2 depth = vec2(0.0);
  for (int i = 0; i < 8; i++) {
    if (i >= steps) break;
    vec3 s = p + dir * (len * (float(i) + 0.5));
    float h = max(length(s) - a.planetR, 0.0);
    depth += vec2(atmoDensityR(a, h), atmoDensityM(a, h)) * len;
  }
  return depth;
}

// Integrates in-scattered light along the segment [t0, t1] of a ray that starts
// at the camera (origin) travelling along rd, in the planet's local frame where
// the planet centre is given by the centre argument.
vec3 scatter(Atmo a, vec3 rd, vec3 centre, float t0, float t1, vec3 sunDir,
             vec3 sunColour, int viewSteps, out float outTransmittance) {
  float len = (t1 - t0) / float(viewSteps);
  vec2 depthView = vec2(0.0);
  vec3 sumR = vec3(0.0), sumM = vec3(0.0);

  for (int i = 0; i < 24; i++) {
    if (i >= viewSteps) break;
    vec3 p = rd * (t0 + len * (float(i) + 0.5)) - centre;   // relative to planet
    float h = max(length(p) - a.planetR, 0.0);
    vec2 d = vec2(atmoDensityR(a, h), atmoDensityM(a, h)) * len;
    depthView += d;

    // Is this sample in the planet's shadow?
    vec2 shadow = raySphere(sunDir, -p, a.planetR);
    float lit = (shadow.x > 0.0) ? 0.0 : 1.0;
    if (lit > 0.0) {
      vec2 depthSun = opticalDepthLight(a, p, sunDir, 5);
      vec3 tau = a.betaR * (depthSun.x + depthView.x) + vec3(a.betaM * 1.1) * (depthSun.y + depthView.y);
      vec3 att = exp(-tau);
      sumR += att * d.x;
      sumM += att * d.y;
    }
  }

  float mu = dot(rd, sunDir);
  float phaseR = 3.0 / (16.0 * 3.14159265) * (1.0 + mu * mu);
  float g2 = a.g * a.g;
  float phaseM = 3.0 / (8.0 * 3.14159265) * ((1.0 - g2) * (1.0 + mu * mu)) /
                 ((2.0 + g2) * pow(1.0 + g2 - 2.0 * a.g * mu, 1.5));

  outTransmittance = exp(-(a.betaR.g * depthView.x + a.betaM * depthView.y));
  return sunColour * (sumR * a.betaR * phaseR + sumM * a.betaM * phaseM);
}
`;

/* ---------------------------------------------------------------- planet -- */

const PLANET_FS = `#version 300 es
precision highp float;
in vec3 vRay;
out vec4 fragColor;

uniform vec3 uCentre;
uniform float uRadius;
uniform float uAtmoScale;      // atmosphere thickness in planet radii
uniform vec3 uSunDir;          // planet -> star, normalised
uniform vec3 uSunColour;       // colour times irradiance
uniform vec3 uAxis;
uniform float uSpin;
uniform float uSeed;
uniform int uType;
uniform float uTemperature;
uniform float uCloudiness;
uniform float uOceanLevel;
uniform float uIceLatitude;
uniform float uRingInner;
uniform float uRingOuter;
uniform float uRingOpacity;
uniform vec3 uRingAxis;
uniform float uRingSeed;
uniform float uTime;
uniform float uHabitable;
uniform float uExposure;

${HASH_GLSL}
${BLACKBODY_GLSL}
${SPHERE_GLSL}
${ATMOSPHERE_GLSL}

// Ridged noise makes mountain chains rather than rolling hills.
float ridged(vec3 p, int oct) {
  float a = 0.5, s = 0.0, n = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= oct) break;
    s += a * (1.0 - abs(vnoise(p)));
    n += a; a *= 0.5; p *= 2.03;
  }
  return s / n;
}

// Surface albedo and emission for a point on the unit sphere in body frame.
void surface(vec3 n, out vec3 albedo, out vec3 emissive, out float gloss) {
  float seed = uSeed;
  vec3 p = n * 2.6 + vec3(seed * 0.37, seed * 0.11, seed * 0.73);
  float lat = abs(n.z);                       // body frame: z is the pole

  emissive = vec3(0.0);
  gloss = 0.0;

  if (uType == 7 || uType == 6) {
    // --- giants: zonal bands sheared by differential rotation --------
    // Band count and palette vary with the seed: some worlds come out like
    // Jupiter's browns and creams, others like Saturn's flat gold or the deep
    // blue of an ice giant whose methane absorbs the red.
    float nBands = 7.0 + 8.0 * fract(seed * 0.137);
    float band = n.z * nBands + 1.8 * fbm(vec3(p.xy * 1.4, n.z * 5.0), 4);
    float t = 0.5 + 0.5 * sin(band);
    t = smoothstep(0.18, 0.82, t);          // crisper zone/belt boundaries
    float hue = fract(seed * 0.311);
    vec3 c1, c2;
    if (uType == 7) {
      c1 = mix(vec3(0.92, 0.84, 0.68), vec3(0.90, 0.72, 0.42), hue);
      c2 = mix(vec3(0.52, 0.33, 0.20), vec3(0.66, 0.46, 0.24), hue);
    } else {
      c1 = mix(vec3(0.48, 0.74, 0.90), vec3(0.36, 0.60, 0.88), hue);
      c2 = mix(vec3(0.16, 0.34, 0.66), vec3(0.22, 0.46, 0.74), hue);
    }
    albedo = mix(c1, c2, t);
    // Storms: long-lived anticyclones stretched by the zonal flow.
    float storm = fbm(vec3(p.x * 2.2, p.y * 2.2, p.z * 6.0), 4);
    float sm = smoothstep(0.34, 0.62, storm) * (1.0 - abs(n.z));
    albedo = mix(albedo, uType == 7 ? vec3(0.92, 0.55, 0.36) : vec3(0.72, 0.86, 0.95), sm * 0.8);
    albedo *= 0.9 + 0.2 * fbm(p * 6.0, 3);
    return;
  }

  float h = fbm(p, 6) * 0.6 + ridged(p * 1.7, 5) * 0.4;
  float continents = fbm(p * 0.55 + 31.0, 4);

  if (uType == 0) {
    // --- lava: cooled basalt cracked open over a glowing interior ----
    float crack = 1.0 - abs(fbm(p * 2.4, 5));
    crack = pow(smoothstep(0.55, 0.95, crack), 2.0);
    albedo = mix(vec3(0.10, 0.075, 0.07), vec3(0.22, 0.15, 0.12), h);
    emissive = blackbodyRGB(clamp(uTemperature * 1.35, 1200.0, 3200.0)) * crack * 3.2;
    return;
  }
  if (uType == 5) {
    // --- ice: fractured shell over a darker subsurface ---------------
    float frac = 1.0 - abs(fbm(p * 3.1, 5));
    albedo = mix(vec3(0.80, 0.86, 0.94), vec3(0.55, 0.66, 0.80), smoothstep(0.5, 0.95, frac));
    gloss = 0.25;
    return;
  }
  if (uType == 1) {
    // --- bare rock, cratered -----------------------------------------
    float crater = smoothstep(0.62, 0.72, abs(fbm(p * 4.5, 4)));
    albedo = mix(vec3(0.28, 0.26, 0.24), vec3(0.42, 0.40, 0.37), h);
    albedo = mix(albedo, vec3(0.20, 0.19, 0.18), crater * 0.6);
    return;
  }
  if (uType == 2) {
    // --- desert: dunes and oxidised iron -----------------------------
    float dune = 0.5 + 0.5 * sin(dot(p, vec3(9.0, 3.0, 5.0)) + fbm(p * 2.0, 3) * 6.0);
    albedo = mix(vec3(0.62, 0.42, 0.24), vec3(0.80, 0.62, 0.38), dune * 0.5 + h * 0.5);
    albedo = mix(albedo, vec3(0.86, 0.84, 0.80), smoothstep(uIceLatitude, uIceLatitude + 0.12, lat) * 0.5);
    return;
  }

  // --- terran and ocean worlds ---------------------------------------
  float sea = uOceanLevel;
  float land = smoothstep(sea - 0.02, sea + 0.02, h + continents * 0.35);

  // Latitude drives the biome: ice at the poles, forest at temperate
  // latitudes, desert near the equator where the Hadley cells descend.
  float polar = smoothstep(uIceLatitude - 0.08, uIceLatitude + 0.06, lat);
  float arid = smoothstep(0.30, 0.02, lat) * smoothstep(0.35, 0.65, fbm(p * 1.3 + 7.0, 3) + 0.5);

  vec3 forest = vec3(0.10, 0.26, 0.11);
  vec3 grass = vec3(0.28, 0.34, 0.16);
  vec3 desert = vec3(0.66, 0.54, 0.33);
  vec3 rock = vec3(0.34, 0.31, 0.28);
  vec3 snow = vec3(0.90, 0.93, 0.96);

  vec3 ground = mix(forest, grass, smoothstep(0.35, 0.65, fbm(p * 3.1, 4) + 0.5));
  ground = mix(ground, desert, arid);
  ground = mix(ground, rock, smoothstep(sea + 0.22, sea + 0.42, h));
  ground = mix(ground, snow, polar);

  vec3 deep = vec3(0.012, 0.045, 0.11);
  vec3 shallow = vec3(0.04, 0.20, 0.30);
  float depth = smoothstep(sea - 0.18, sea, h + continents * 0.35);
  vec3 ocean = mix(deep, shallow, depth);
  ocean = mix(ocean, vec3(0.72, 0.84, 0.90), polar * 0.85);   // sea ice

  albedo = mix(ocean, ground, land);
  gloss = (1.0 - land) * (1.0 - polar * 0.8) * 0.85;
}

void main() {
  vec3 rd = normalize(vRay);
  float atmoR = uRadius * (1.0 + uAtmoScale);

  vec2 tp = raySphere(rd, uCentre, uRadius);
  vec2 ta = uAtmoScale > 0.0005 ? raySphere(rd, uCentre, atmoR) : vec2(-1.0);
  if (tp.y < 0.0 && ta.y < 0.0) discard;

  vec3 colour = vec3(0.0);
  float alpha = 0.0;

  /* ---- solid surface ------------------------------------------------ */
  bool hitSurface = tp.x > 0.0;
  if (hitSurface) {
    vec3 hit = rd * tp.x;
    vec3 n = normalize(hit - uCentre);
    mat3 f = bodyFrame(uAxis, uSpin);
    vec3 nb = n * f;                        // into the body frame

    vec3 albedo, emissive; float gloss;
    surface(nb, albedo, emissive, gloss);

    float ndl = dot(n, uSunDir);
    // A soft terminator: the star is not a point source, and the atmosphere
    // scatters light past the geometric limb.
    float diffuse = smoothstep(-0.12, 0.22, ndl);

    vec3 lit = albedo * uSunColour * diffuse;

    // Specular from oceans and ice.
    if (gloss > 0.01) {
      vec3 h = normalize(uSunDir - rd);
      float spec = pow(max(dot(n, h), 0.0), 220.0) * gloss;
      float fres = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
      lit += uSunColour * spec * (0.35 + fres) * 2.4;
    }

    // Clouds, advected by the body's rotation.
    if (uCloudiness > 0.01 && uType < 6) {
      vec3 cp = nb * 3.4 + vec3(uTime * 0.006 + uSeed, uSeed * 0.5, 0.0);
      float cloud = fbm(cp, 5) * 0.5 + 0.5;
      cloud = smoothstep(1.0 - uCloudiness * 0.85, 1.0 - uCloudiness * 0.85 + 0.22, cloud);
      lit = mix(lit, uSunColour * 0.92 * diffuse, cloud * 0.85);
    }

    // Night side: thermal glow on lava worlds, and on a habitable world the
    // faint suggestion of something looking back.
    lit += emissive * albedo * 0.0 + emissive;
    if (uHabitable > 0.5) {
      float night = smoothstep(0.06, -0.24, ndl);
      vec3 lp = nb * 9.0 + uSeed;
      float city = smoothstep(0.62, 0.80, fbm(lp, 4) + 0.5);
      float landMask = step(0.35, dot(albedo, vec3(0.33)));
      lit += vec3(1.0, 0.82, 0.52) * city * night * landMask * 0.05;
    }

    colour += lit;
    alpha = 1.0;
  }

  /* ---- rings, if the ray crosses the ring plane in front ------------- */
  if (uRingOpacity > 0.001) {
    vec3 rn = normalize(uRingAxis);
    float denom = dot(rd, rn);
    if (abs(denom) > 1e-6) {
      float tr = dot(uCentre, rn) / denom;
      if (tr > 0.0 && (!hitSurface || tr < tp.x)) {
        vec3 hp = rd * tr - uCentre;
        float r = length(hp);
        if (r > uRingInner && r < uRingOuter) {
          float u = (r - uRingInner) / (uRingOuter - uRingInner);
          // Radial structure: resonances with the moons carve gaps, and the
          // particles themselves clump into fine ringlets.
          float fine = 0.5 + 0.5 * sin(u * 220.0 + uRingSeed * 13.0);
          float mid = 0.5 + 0.5 * sin(u * 41.0 + uRingSeed * 3.0);
          float gap = smoothstep(0.02, 0.06, abs(fract(u * 3.0 + 0.21) - 0.5));
          float dens = uRingOpacity * gap * (0.55 + 0.30 * mid + 0.15 * fine)
                     * smoothstep(0.0, 0.06, u) * (1.0 - smoothstep(0.85, 1.0, u));

          // Is this piece of the ring in the planet's shadow?
          vec2 sh = raySphere(uSunDir, -hp, uRadius);
          float lit = sh.x > 0.0 ? 0.06 : 1.0;

          // Ring particles are icy and strongly forward-scattering.
          float mu = dot(rd, uSunDir);
          float fwd = 0.35 + 0.65 * pow(max(mu, 0.0), 6.0);
          vec3 ringCol = mix(vec3(0.78, 0.74, 0.66), vec3(0.92, 0.90, 0.86), fine);
          // Ring particles are icy and highly reflective; a Bond albedo near
          // 0.5 with strong forward scattering is what makes Saturn's rings
          // comparable in brightness to the planet itself.
          vec3 ringLit = ringCol * uSunColour * lit * (0.35 + fwd) * 1.8;

          // Slant path: a grazing view crosses more material.
          float slant = clamp(dens / max(abs(denom), 0.02), 0.0, 1.0);
          colour = colour * (1.0 - slant) + ringLit * slant;
          alpha = max(alpha, slant);
        }
      }
    }
  }

  /* ---- atmosphere ---------------------------------------------------- */
  if (ta.y > 0.0 && uAtmoScale > 0.0005) {
    Atmo a;
    a.planetR = uRadius;
    a.atmoR = atmoR;
    a.hRayleigh = uRadius * uAtmoScale * 0.28;
    a.hMie = uRadius * uAtmoScale * 0.09;
    // Rayleigh scattering goes as the inverse fourth power of wavelength; the
    // ratio below is that law evaluated at 680, 550 and 440 nanometres.
    float density = 5.2 / max(uAtmoScale, 1e-4) * 0.02;
    a.betaR = vec3(0.1735, 0.4045, 0.9911) * density;
    a.betaM = 0.06 * density;
    a.g = 0.76;

    float t0 = max(ta.x, 0.0);
    float t1 = hitSurface ? min(ta.y, tp.x) : ta.y;
    if (t1 > t0) {
      float trans;
      vec3 sky = scatter(a, rd, uCentre, t0, t1, uSunDir, uSunColour, 14, trans);
      colour = colour * trans + sky;
      alpha = max(alpha, clamp(1.0 - trans, 0.0, 1.0));
    }
  }

  if (alpha < 0.002) discard;
  fragColor = vec4(colour * uExposure, alpha);
}`;

/* ------------------------------------------------------------------ star -- */

const STAR_FS = `#version 300 es
precision highp float;
in vec3 vRay;
out vec4 fragColor;
uniform vec3 uCentre;
uniform float uRadius;
uniform float uTemperature;
uniform float uLuminosity;   // solar luminosities
uniform float uSurface;      // exposure of the resolved photosphere
uniform float uGlare;        // exposure of the unresolved glare
uniform float uTime;
uniform float uSeed;
${HASH_GLSL}
${BLACKBODY_GLSL}
${SPHERE_GLSL}

void main() {
  vec3 rd = normalize(vRay);
  float d = length(uCentre);
  vec2 t = raySphere(rd, uCentre, uRadius);

  vec3 base = blackbodyRGB(uTemperature);
  vec3 colour = vec3(0.0);

  if (t.x > 0.0) {
    vec3 n = normalize(rd * t.x - uCentre);
    float mu = max(dot(n, -rd), 0.0);
    // Limb darkening: a line of sight at the edge of the disc reaches only the
    // cooler upper photosphere. Eddington's linear law.
    float limb = 0.35 + 0.65 * mu;
    // Granulation: the tops of convection cells, a few percent in contrast.
    float gran = fbm(n * 26.0 + vec3(uTime * 0.04, uSeed, 0.0), 4);
    float faculae = smoothstep(0.25, 0.6, gran);
    // Surface brightness follows Stefan-Boltzmann, so it does not depend on
    // distance at all - only on temperature. A 3000 K red dwarf's photosphere
    // really is around thirty times dimmer per unit area than the Sun's, and
    // that is why it looks like an ember rather than a spotlight.
    float sb = pow(uTemperature / 5772.0, 4.0);
    colour = base * uSurface * sb * limb * (0.94 + 0.16 * faculae);
    colour = mix(colour, vec3(1.0), pow(mu, 8.0) * 0.35);
  }

  // Glare from the unresolved star. Its strength is the flux actually arriving,
  // L / d^2, in units of the Sun's at one astronomical unit - which is why a
  // dim star still blazes when you are close to it and a bright one fades when
  // you are far away.
  float flux = uLuminosity / max(d * d, 1e-12);
  float ang = length(cross(rd, normalize(uCentre)));
  float angR = uRadius / max(d, 1e-9);
  float halo = angR / max(ang, angR * 0.35);
  float profile = pow(halo, 2.4) * 0.55 + pow(halo, 6.0) * 0.9;
  colour += base * uGlare * flux * profile;

  if (dot(colour, vec3(1.0)) < 1e-6) discard;
  fragColor = vec4(colour, 1.0);
}`;

/* --------------------------------------------------------------- debris -- */

const BELT_VS = `#version 300 es
precision highp float;
uniform mat4 uViewProjRel;
uniform vec3 uCameraPos;      // relative to the system barycentre
uniform vec3 uCentre;         // system centre relative to the camera
uniform float uInner;
uniform float uOuter;
uniform float uIncSpread;
uniform float uSeed;
uniform float uTime;          // years
uniform float uStarMass;
uniform float uPixelsPerRadian;
uniform float uSize;
uniform float uBrightness;
uniform vec3 uSunColour;
out vec3 vColour;
out float vIntensity;
${HASH_GLSL}

void main() {
  uint id = uint(gl_VertexID) ^ uint(int(uSeed));
  vec3 h = hash3(id * 2654435761u);
  float h4 = hash1(id * 40503u);

  // Semi-major axis distributed so the surface density falls as 1/r, which is
  // roughly what a collisionally relaxed belt looks like.
  float a = mix(uInner, uOuter, h.x * h.x);
  float e = h4 * 0.14;
  float inc = (h.y - 0.5) * 2.0 * uIncSpread;
  float node = h.z * 6.28318530718;

  // Kepler's third law: the inner edge laps the outer one.
  float period = sqrt(a * a * a / max(uStarMass, 0.02));
  float M = node + 6.28318530718 * uTime / period;

  // Two Newton steps are plenty at these small eccentricities.
  float E = M;
  for (int i = 0; i < 3; i++) E -= (E - e * sin(E) - M) / (1.0 - e * cos(E));

  float x = a * (cos(E) - e);
  float y = a * sqrt(1.0 - e * e) * sin(E);
  float ci = cos(inc), si = sin(inc);
  float cn = cos(node * 1.7), sn = sin(node * 1.7);
  vec3 pos = vec3(x * cn - y * ci * sn, y * si, x * sn + y * ci * cn);

  vec3 rel = uCentre + pos;
  float dist = length(rel);
  gl_Position = uViewProjRel * vec4(rel, 1.0);

  float sizePx = uSize * uPixelsPerRadian / max(dist, 1e-9);
  gl_PointSize = clamp(sizePx, 1.0, 6.0);

  // Illuminated by the star, so the belt fades with distance from it.
  float r = length(pos);
  float irradiance = 1.0 / max(r * r, 1e-6);
  float shrink = min(1.0, sizePx * sizePx);
  vIntensity = uBrightness * irradiance * shrink;
  vColour = mix(vec3(0.62, 0.56, 0.48), vec3(0.80, 0.84, 0.92), h.z) * uSunColour;
  if (vIntensity < 1e-6) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;

const POINT_FS = `#version 300 es
precision highp float;
in vec3 vColour;
in float vIntensity;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  fragColor = vec4(vColour * (vIntensity * exp(-r2 * 2.6)), 1.0);
}`;

/* ----------------------------------------------------------- orbit paths -- */

// A planet's orbit drawn as the ellipse it actually is: the vertex shader walks
// the eccentric anomaly, so the curve is denser near periapsis exactly where
// the planet moves fastest, and an eccentric orbit is visibly off-centre from
// its star.
const ORBIT_VS = `#version 300 es
precision highp float;
uniform mat4 uViewProjRel;
uniform vec3 uCentre;       // the star, relative to the camera
uniform float uA;           // semi-major axis
uniform float uE;
uniform float uInc;
uniform float uNode;
uniform float uPeri;
uniform int uSegments;
out float vT;

void main() {
  float E = 6.28318530718 * float(gl_VertexID) / float(uSegments);
  float x = uA * (cos(E) - uE);
  float y = uA * sqrt(max(1.0 - uE * uE, 0.0)) * sin(E);

  float cw = cos(uPeri), sw = sin(uPeri);
  float cO = cos(uNode), sO = sin(uNode);
  float ci = cos(uInc), si = sin(uInc);
  float x1 = x * cw - y * sw;
  float y1 = x * sw + y * cw;
  vec3 p = vec3(x1 * cO - y1 * ci * sO, y1 * si, x1 * sO + y1 * ci * cO);

  gl_Position = uViewProjRel * vec4(uCentre + p, 1.0);
  vT = float(gl_VertexID) / float(uSegments);
}`;

const ORBIT_FS = `#version 300 es
precision highp float;
in float vT;
out vec4 fragColor;
uniform vec3 uColour;
uniform float uOpacity;
void main() { fragColor = vec4(uColour * uOpacity, 1.0); }`;

/* ---------------------------------------------------------------- driver -- */

export class BodyRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.progPlanet = ctx.program(IMPOSTOR_VS, PLANET_FS, 'planet');
    this.progStar = ctx.program(IMPOSTOR_VS, STAR_FS, 'star');
    this.progBelt = ctx.program(BELT_VS, POINT_FS, 'belt');
    this.progOrbit = ctx.program(ORBIT_VS, ORBIT_FS, 'orbit-path');
    this.vao = ctx.gl.createVertexArray();
    this._vp = m4();
    this._tmp = v3();
  }

  begin(camera) {
    const gl = this.gl;
    const viewRel = m4();
    camera.viewNoTranslation(viewRel);
    m4mul(this._vp, camera.proj, viewRel);
    this.pixelsPerRadian = (gl.drawingBufferHeight * 0.5) / Math.tan(camera.fov * 0.5);
    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
  }

  // Bodies are drawn with premultiplied source-over so an atmosphere can sit in
  // front of whatever is behind it, then the star and debris add on top.
  _alphaBlend() {
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }
  _addBlend() {
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
  }

  drawStar(camera, { centre, radius, temperature, luminosity, surface, glare, time, seed }) {
    this._addBlend();
    const p = this.progStar.use()
      .set('uViewProjRel', this._vp)
      .set('uCamRight', camera.right)
      .set('uCamUp', camera.up)
      .set('uCentre', centre)
      .set('uRadius', radius)
      // The quad must be large enough to hold the glare, not just the disc.
      .set('uQuadRadius', Math.max(radius * 26, Math.hypot(centre[0], centre[1], centre[2]) * 0.34))
      .set('uTemperature', temperature)
      .set('uLuminosity', luminosity)
      .set('uSurface', surface)
      .set('uGlare', glare)
      .set('uTime', time)
      .set('uSeed', seed);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.ctx.drawCalls++;
  }

  drawPlanet(camera, o) {
    this._alphaBlend();
    const quad = Math.max(
      o.radius * (1 + (o.atmoScale || 0)) * 1.06,
      o.ringOuter ? o.ringOuter * 1.02 : 0,
    );
    const p = this.progPlanet.use()
      .set('uViewProjRel', this._vp)
      .set('uCamRight', camera.right)
      .set('uCamUp', camera.up)
      .set('uCentre', o.centre)
      .set('uRadius', o.radius)
      .set('uQuadRadius', quad)
      .set('uAtmoScale', o.atmoScale || 0)
      .set('uSunDir', o.sunDir)
      .set('uSunColour', o.sunColour)
      .set('uAxis', o.axis)
      .set('uSpin', o.spin || 0)
      .set('uSeed', o.seed)
      .set('uType', o.type)
      .set('uTemperature', o.temperature || 280)
      .set('uCloudiness', o.cloudiness || 0)
      .set('uOceanLevel', o.oceanLevel ?? 0.5)
      .set('uIceLatitude', o.iceLatitude ?? 0.82)
      .set('uRingInner', o.ringInner || 0)
      .set('uRingOuter', o.ringOuter || 0)
      .set('uRingOpacity', o.ringOpacity || 0)
      .set('uRingAxis', o.ringAxis || o.axis)
      .set('uRingSeed', o.ringSeed || 0)
      .set('uHabitable', o.habitable ? 1 : 0)
      .set('uExposure', o.exposure ?? 1)
      .set('uTime', o.time || 0);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.ctx.drawCalls++;
  }

  drawBelt(camera, o) {
    this._addBlend();
    const p = this.progBelt.use()
      .set('uViewProjRel', this._vp)
      .set('uCentre', o.centre)
      .set('uInner', o.inner)
      .set('uOuter', o.outer)
      .set('uIncSpread', o.incSpread)
      .set('uSeed', o.seed % 65536)
      .set('uTime', o.time)
      .set('uStarMass', o.starMass)
      .set('uPixelsPerRadian', this.pixelsPerRadian)
      .set('uSize', o.size)
      .set('uBrightness', o.brightness)
      .set('uSunColour', o.sunColour);
    this.gl.drawArrays(this.gl.POINTS, 0, o.count);
    this.ctx.drawCalls++;
  }

  drawOrbit(camera, o) {
    this._addBlend();
    const segments = o.segments || 256;
    this.progOrbit.use()
      .set('uViewProjRel', this._vp)
      .set('uCentre', o.centre)
      .set('uA', o.semiMajorAU)
      .set('uE', o.eccentricity || 0)
      .set('uInc', o.inclination || 0)
      .set('uNode', o.longitudeAscending || 0)
      .set('uPeri', o.argumentPeriapsis || 0)
      .set('uSegments', segments)
      .set('uColour', o.colour)
      .set('uOpacity', o.opacity);
    this.gl.drawArrays(this.gl.LINE_LOOP, 0, segments);
    this.ctx.drawCalls++;
  }

  end() {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }
}
