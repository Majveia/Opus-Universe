// The HDR resolve chain. Everything upstream renders unbounded radiance into a
// float buffer; this file turns that into pixels: energy-conserving bloom, an
// ACES filmic transform, and dithering tuned for OLED panels where a banded
// near-black gradient is glaringly visible.

import { FULLSCREEN_VS, PingPong } from '../core/gl.js';

const COMMON = `
const float PI = 3.14159265359;

// Rec. 709 luminance.
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Interleaved gradient noise - cheap, temporally stable, and visually neutral.
float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}
`;

const DOWNSAMPLE_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2 uTexel;      // 1 / source resolution
uniform float uFirstPass; // Karis-average only on the first tap to kill fireflies
${COMMON}

vec3 tap(vec2 uv) { return texture(uSrc, uv).rgb; }

// Weighted 13-tap box grid from Jimenez, "Next Generation Post Processing".
void main() {
  vec2 t = uTexel;
  vec3 a = tap(vUV + t * vec2(-2.0,  2.0));
  vec3 b = tap(vUV + t * vec2( 0.0,  2.0));
  vec3 c = tap(vUV + t * vec2( 2.0,  2.0));
  vec3 d = tap(vUV + t * vec2(-2.0,  0.0));
  vec3 e = tap(vUV);
  vec3 f = tap(vUV + t * vec2( 2.0,  0.0));
  vec3 g = tap(vUV + t * vec2(-2.0, -2.0));
  vec3 h = tap(vUV + t * vec2( 0.0, -2.0));
  vec3 i = tap(vUV + t * vec2( 2.0, -2.0));
  vec3 j = tap(vUV + t * vec2(-1.0,  1.0));
  vec3 k = tap(vUV + t * vec2( 1.0,  1.0));
  vec3 l = tap(vUV + t * vec2(-1.0, -1.0));
  vec3 m = tap(vUV + t * vec2( 1.0, -1.0));

  vec3 result;
  if (uFirstPass > 0.5) {
    // Average each 2x2 group in inverse-luma weighted space, which stops a
    // single very bright pixel from dominating the whole bloom kernel.
    vec3 g0 = (j + k + l + m) * 0.25;
    vec3 g1 = (a + b + d + e) * 0.25;
    vec3 g2 = (b + c + e + f) * 0.25;
    vec3 g3 = (d + e + g + h) * 0.25;
    vec3 g4 = (e + f + h + i) * 0.25;
    float w0 = 1.0 / (1.0 + luma(g0));
    float w1 = 1.0 / (1.0 + luma(g1));
    float w2 = 1.0 / (1.0 + luma(g2));
    float w3 = 1.0 / (1.0 + luma(g3));
    float w4 = 1.0 / (1.0 + luma(g4));
    float wsum = w0 * 0.5 + (w1 + w2 + w3 + w4) * 0.125;
    result = (g0 * w0 * 0.5 + g1 * w1 * 0.125 + g2 * w2 * 0.125 + g3 * w3 * 0.125 + g4 * w4 * 0.125) / max(wsum, 1e-5);
  } else {
    result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
  }
  fragColor = vec4(max(result, vec3(0.0)), 1.0);
}`;

const UPSAMPLE_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uSrc;    // smaller mip being spread
uniform vec2 uTexel;
uniform float uRadius;
${COMMON}

// 3x3 tent filter - the standard energy-preserving partner to the box down-tap.
void main() {
  vec2 t = uTexel * uRadius;
  vec3 s = texture(uSrc, vUV + t * vec2(-1.0,  1.0)).rgb * 1.0;
  s += texture(uSrc, vUV + t * vec2( 0.0,  1.0)).rgb * 2.0;
  s += texture(uSrc, vUV + t * vec2( 1.0,  1.0)).rgb * 1.0;
  s += texture(uSrc, vUV + t * vec2(-1.0,  0.0)).rgb * 2.0;
  s += texture(uSrc, vUV                       ).rgb * 4.0;
  s += texture(uSrc, vUV + t * vec2( 1.0,  0.0)).rgb * 2.0;
  s += texture(uSrc, vUV + t * vec2(-1.0, -1.0)).rgb * 1.0;
  s += texture(uSrc, vUV + t * vec2( 0.0, -1.0)).rgb * 2.0;
  s += texture(uSrc, vUV + t * vec2( 1.0, -1.0)).rgb * 1.0;
  fragColor = vec4(s * (1.0 / 16.0), 1.0);
}`;

const COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform vec2 uResolution;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uVignette;
uniform float uGrain;
uniform float uChroma;
uniform float uTime;
uniform float uSaturation;
uniform float uFade;       // cinematic fade-to-black for scale transitions
${COMMON}

// ACES filmic, fitted by Stephen Hill. Desaturating highlights the way real
// film does is what stops star cores from clipping to flat magenta.
const mat3 ACESInput = mat3(
  0.59719, 0.07600, 0.02840,
  0.35458, 0.90834, 0.13383,
  0.04823, 0.01566, 0.83777);
const mat3 ACESOutput = mat3(
   1.60475, -0.10208, -0.00327,
  -0.53108,  1.10813, -0.07276,
  -0.07367, -0.00605,  1.07602);

vec3 RRTAndODTFit(vec3 v) {
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return a / b;
}

vec3 acesFitted(vec3 color) {
  color = ACESInput * color;
  color = RRTAndODTFit(color);
  color = ACESOutput * color;
  return clamp(color, 0.0, 1.0);
}

vec3 sampleSceneChroma(vec2 uv, float amount) {
  if (amount < 1e-4) return texture(uScene, uv).rgb;
  // Lateral chromatic aberration grows toward the frame edge, like real glass.
  vec2 d = uv - 0.5;
  float r2 = dot(d, d);
  vec2 off = d * r2 * amount;
  return vec3(
    texture(uScene, uv + off * 1.0).r,
    texture(uScene, uv).g,
    texture(uScene, uv - off * 1.0).b);
}

void main() {
  vec3 scene = sampleSceneChroma(vUV, uChroma);
  vec3 bloom = texture(uBloom, vUV).rgb;
  vec3 color = scene + bloom * uBloomStrength;

  color *= uExposure;
  color = acesFitted(color);

  // Saturation trim in perceptual space, after tonemapping.
  float l = luma(color);
  color = mix(vec3(l), color, uSaturation);

  // Vignette: a soft cos^4 falloff, never a hard ring.
  vec2 d = (vUV - 0.5) * vec2(uResolution.x / uResolution.y, 1.0);
  float vig = 1.0 - uVignette * dot(d, d) * 0.85;
  color *= clamp(vig, 0.0, 1.0);

  // Film grain, scaled by (1-luma) so shadows stay alive but highlights stay clean.
  float n = ign(gl_FragCoord.xy + vec2(uTime * 61.0, uTime * 37.0)) - 0.5;
  color += n * uGrain * (1.0 - smoothstep(0.0, 0.8, luma(color)));

  color *= uFade;

  // Ordered dither before the 8-bit quantise. Without this, the deep-space
  // gradients that dominate this scene band into visible contour rings on OLED.
  float dither = (ign(gl_FragCoord.xy) - 0.5) / 255.0;
  color += dither;

  fragColor = vec4(max(color, vec3(0.0)), 1.0);
}`;

const FXAA_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2 uTexel;
${COMMON}

// Compact FXAA 3.11 (quality preset trimmed for a scene dominated by points
// and gradients rather than hard polygon edges).
void main() {
  vec3 rgbM = texture(uSrc, vUV).rgb;
  vec3 rgbNW = texture(uSrc, vUV + uTexel * vec2(-1.0, -1.0)).rgb;
  vec3 rgbNE = texture(uSrc, vUV + uTexel * vec2( 1.0, -1.0)).rgb;
  vec3 rgbSW = texture(uSrc, vUV + uTexel * vec2(-1.0,  1.0)).rgb;
  vec3 rgbSE = texture(uSrc, vUV + uTexel * vec2( 1.0,  1.0)).rgb;

  float lNW = luma(rgbNW), lNE = luma(rgbNE), lSW = luma(rgbSW), lSE = luma(rgbSE), lM = luma(rgbM);
  float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

  if (lMax - lMin < max(0.0312, lMax * 0.125)) { fragColor = vec4(rgbM, 1.0); return; }

  vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
  float dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, -8.0, 8.0) * uTexel;

  vec3 rgbA = 0.5 * (texture(uSrc, vUV + dir * (1.0 / 3.0 - 0.5)).rgb +
                     texture(uSrc, vUV + dir * (2.0 / 3.0 - 0.5)).rgb);
  vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(uSrc, vUV - dir * 0.5).rgb +
                                   texture(uSrc, vUV + dir * 0.5).rgb);
  float lB = luma(rgbB);
  fragColor = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}`;

const COPY_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uSrc;
void main() { fragColor = texture(uSrc, vUV); }`;

export class PostChain {
  constructor(ctx, width, height, opts = {}) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.mipCount = opts.mipCount || 7;
    this.settings = {
      exposure: 1.0,
      bloomStrength: 0.72,
      bloomRadius: 1.0,
      vignette: 0.42,
      grain: 0.018,
      chroma: 0.22,
      saturation: 1.06,
      fade: 1.0,
      fxaa: true,
    };

    this.progDown = ctx.program(FULLSCREEN_VS, DOWNSAMPLE_FS, 'bloom-down');
    this.progUp = ctx.program(FULLSCREEN_VS, UPSAMPLE_FS, 'bloom-up');
    this.progComposite = ctx.program(FULLSCREEN_VS, COMPOSITE_FS, 'composite');
    this.progFXAA = ctx.program(FULLSCREEN_VS, FXAA_FS, 'fxaa');
    this.progCopy = ctx.program(FULLSCREEN_VS, COPY_FS, 'copy');

    this.resize(width, height);
  }

  resize(width, height) {
    const gl = this.gl;
    this.width = Math.max(1, width | 0);
    this.height = Math.max(1, height | 0);
    this.dispose(false);

    const mk = (w, h) => {
      const tex = this.ctx.texture({
        width: Math.max(1, w), height: Math.max(1, h),
        format: 'RGBA16F', filter: gl.LINEAR, wrap: gl.CLAMP_TO_EDGE,
      });
      return { tex, fbo: this.ctx.framebuffer(tex), width: Math.max(1, w), height: Math.max(1, h) };
    };

    // Depth is stored separately so scene passes can depth-test against it.
    this.sceneDepth = this.ctx.texture({
      width: this.width, height: this.height, format: 'DEPTH_COMPONENT32F',
      filter: gl.NEAREST, wrap: gl.CLAMP_TO_EDGE,
    });
    this.sceneTex = this.ctx.texture({
      width: this.width, height: this.height, format: 'RGBA16F', filter: gl.LINEAR, wrap: gl.CLAMP_TO_EDGE,
    });
    this.sceneFBO = this.ctx.framebuffer(this.sceneTex, this.sceneDepth);

    this.mips = [];
    let w = this.width, h = this.height;
    for (let i = 0; i < this.mipCount; i++) {
      w = Math.max(1, w >> 1); h = Math.max(1, h >> 1);
      this.mips.push(mk(w, h));
      if (w === 1 && h === 1) break;
    }
    this.ldrA = mk(this.width, this.height);
  }

  get sceneFramebuffer() { return this.sceneFBO; }

  // Runs bloom + tonemap + AA and presents to the default framebuffer.
  render(time, targetWidth, targetHeight) {
    const gl = this.gl;
    const s = this.settings;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    /* ---- progressive downsample -------------------------------------- */
    let src = this.sceneTex, srcW = this.width, srcH = this.height;
    for (let i = 0; i < this.mips.length; i++) {
      const m = this.mips[i];
      m.fbo.bind();
      this.progDown.use()
        .set('uTexel', new Float32Array([1 / srcW, 1 / srcH]))
        .set('uFirstPass', i === 0 ? 1 : 0)
        .tex('uSrc', src);
      this.ctx.drawFullscreen();
      src = m.tex; srcW = m.width; srcH = m.height;
    }

    /* ---- upsample and accumulate ------------------------------------- */
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = this.mips.length - 1; i > 0; i--) {
      const from = this.mips[i], to = this.mips[i - 1];
      to.fbo.bind();
      this.progUp.use()
        .set('uTexel', new Float32Array([1 / from.width, 1 / from.height]))
        .set('uRadius', s.bloomRadius)
        .tex('uSrc', from.tex);
      this.ctx.drawFullscreen();
    }
    gl.disable(gl.BLEND);

    /* ---- composite ---------------------------------------------------- */
    const compositeTarget = s.fxaa ? this.ldrA.fbo : null;
    if (compositeTarget) compositeTarget.bind();
    else this.ctx.bindDefaultFramebuffer(targetWidth, targetHeight);

    this.progComposite.use()
      .set('uResolution', new Float32Array([this.width, this.height]))
      .set('uExposure', s.exposure)
      .set('uBloomStrength', s.bloomStrength)
      .set('uVignette', s.vignette)
      .set('uGrain', s.grain)
      .set('uChroma', s.chroma)
      .set('uSaturation', s.saturation)
      .set('uFade', s.fade)
      .set('uTime', time)
      .tex('uScene', this.sceneTex)
      .tex('uBloom', this.mips[0].tex);
    this.ctx.drawFullscreen();

    if (s.fxaa) {
      this.ctx.bindDefaultFramebuffer(targetWidth, targetHeight);
      this.progFXAA.use()
        .set('uTexel', new Float32Array([1 / this.width, 1 / this.height]))
        .tex('uSrc', this.ldrA.tex);
      this.ctx.drawFullscreen();
    }
  }

  dispose(full = true) {
    const kill = (o) => { if (o) { o.fbo?.dispose(); o.tex?.dispose(); } };
    if (this.mips) this.mips.forEach(kill);
    kill(this.ldrA);
    if (this.sceneFBO) { this.sceneFBO.dispose(); this.sceneTex.dispose(); this.sceneDepth.dispose(); }
    this.mips = null; this.ldrA = null; this.sceneFBO = null;
  }
}
