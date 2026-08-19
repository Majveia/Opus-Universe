// A three-dimensional complex FFT that runs entirely on the GPU, over a grid
// packed into a 2D atlas.
//
// Uses the Stockham auto-sort formulation, rewritten as a gather so each output
// texel computes its own value from two inputs - which is exactly what a
// fragment shader can do. No bit-reversal pass is needed.
//
// For output index o in a pass of stride Ns:
//     q  = o / (2 Ns)          block
//     r  = o mod Ns            position within the butterfly
//     b  = (o / Ns) mod 2      which half of the butterfly
//     t  = q Ns + r            index of the input pair
//     w  = exp(-i pi r / Ns)   twiddle (conjugated for the inverse)
//     out = in[t] +/- w * in[t + N/2]

import { FULLSCREEN_VS } from '../core/gl.js';
import { ATLAS_GLSL } from './atlas.js';

const FFT_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec2 fragColor;
uniform sampler2D uSrc;
uniform int uAxis;        // 0 = x, 1 = y, 2 = z
uniform int uNs;          // butterfly stride for this pass
uniform float uInverse;   // 0 = forward, 1 = inverse
uniform float uScale;     // 1/N applied on the final inverse pass
uniform sampler2D uTwiddle;  // exact twiddle factors, computed in double on the CPU
${ATLAS_GLSL}

vec2 fetchCell(ivec3 c) {
  return texelFetch(uSrc, cellToTexel(c), 0).xy;
}

void main() {
  int N = int(uAtlas.x);
  ivec3 cell = texelToCell(ivec2(gl_FragCoord.xy));

  int o = uAxis == 0 ? cell.x : (uAxis == 1 ? cell.y : cell.z);
  int Ns2 = uNs * 2;
  int q = o / Ns2;
  int r = o - (o / uNs) * uNs;        // o mod Ns
  int b = (o / uNs) - ((o / uNs) / 2) * 2;   // (o / Ns) mod 2
  int t = q * uNs + r;

  ivec3 c0 = cell, c1 = cell;
  if (uAxis == 0)      { c0.x = t; c1.x = t + N / 2; }
  else if (uAxis == 1) { c0.y = t; c1.y = t + N / 2; }
  else                 { c0.z = t; c1.z = t + N / 2; }

  vec2 in0 = fetchCell(c0);
  vec2 in1 = fetchCell(c1);

  // The angle is always -pi * r / Ns with Ns a power of two, so every twiddle
  // the whole transform ever needs is one of N/2 tabulated values, reached by
  // exact integer indexing. Evaluating sin/cos in the shader instead costs
  // three orders of magnitude of accuracy on drivers with fast transcendentals.
  int j = r * (N / (2 * uNs));
  vec2 w = texelFetch(uTwiddle, ivec2(j, 0), 0).xy;   // already exp(-i pi r / Ns)
  if (uInverse > 0.5) w.y = -w.y;                     // conjugate for the inverse
  vec2 wi1 = vec2(w.x * in1.x - w.y * in1.y, w.x * in1.y + w.y * in1.x);

  fragColor = (b == 0 ? in0 + wi1 : in0 - wi1) * uScale;
}`;

// Loads a scalar real field into the complex working buffer.
const LOAD_FS = `#version 300 es
precision highp float;
out vec2 fragColor;
uniform sampler2D uSrc;
uniform float uOffset;   // subtracted before transforming (mean density)
uniform float uScale;
void main() {
  float v = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).x;
  fragColor = vec2((v - uOffset) * uScale, 0.0);
}`;

export class GPUFFT3D {
  constructor(ctx, layout) {
    const gl = ctx.gl;
    this.ctx = ctx;
    this.gl = gl;
    this.layout = layout;
    this.n = layout.n;
    this.passes = Math.log2(this.n) | 0;

    this.progFFT = ctx.program(FULLSCREEN_VS, FFT_FS, 'gpu-fft');
    this.progLoad = ctx.program(FULLSCREEN_VS, LOAD_FS, 'fft-load');

    const opts = {
      width: layout.width, height: layout.height,
      format: 'RG32F', filter: gl.NEAREST, wrap: gl.CLAMP_TO_EDGE,
    };
    this.a = { tex: ctx.texture(opts) };
    this.b = { tex: ctx.texture(opts) };
    this.a.fbo = ctx.framebuffer(this.a.tex);
    this.b.fbo = ctx.framebuffer(this.b.tex);

    // Twiddle table: exp(-i pi j / (N/2)) for j = 0 .. N/2-1.
    const half = Math.max(1, this.n >> 1);
    const tw = new Float32Array(half * 2);
    for (let j = 0; j < half; j++) {
      const ang = -Math.PI * j / half;
      tw[j * 2] = Math.cos(ang);
      tw[j * 2 + 1] = Math.sin(ang);
    }
    this.twiddle = ctx.texture({
      width: half, height: 1, format: 'RG32F', filter: gl.NEAREST, wrap: gl.CLAMP_TO_EDGE,
    });
    this.twiddle.upload(tw);

    this.atlasUniform = new Float32Array([layout.n, 1 / layout.n, layout.tilesX, layout.tilesY]);
    this.atlasSize = new Float32Array([layout.width, layout.height]);
  }

  _setAtlas(prog) {
    prog.set('uAtlas', this.atlasUniform).set('uAtlasSize', this.atlasSize);
  }

  // Copies a real R32F field into the complex buffer, optionally subtracting a
  // constant (turning a density into a density contrast) and rescaling.
  load(realTex, offset = 0, scale = 1) {
    this.a.fbo.bind();
    this.progLoad.use().set('uOffset', offset).set('uScale', scale).tex('uSrc', realTex);
    this.ctx.drawFullscreen();
  }

  get current() { return this.a.tex; }
  get currentFBO() { return this.a.fbo; }

  _swap() { const t = this.a; this.a = this.b; this.b = t; }

  // Transforms whatever is currently in the working buffer, in place.
  transform(inverse = false) {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    const inv = inverse ? 1 : 0;
    const totalPasses = 3 * this.passes;
    let done = 0;
    for (let axis = 0; axis < 3; axis++) {
      for (let p = 0; p < this.passes; p++) {
        const Ns = 1 << p;
        done++;
        // Fold the 1/N^3 normalisation of the inverse transform into the very
        // last pass, so no separate scaling pass is needed.
        const scale = (inverse && done === totalPasses) ? 1 / (this.n * this.n * this.n) : 1;
        this.b.fbo.bind();
        this.progFFT.use()
          .set('uAxis', axis).set('uNs', Ns).set('uInverse', inv).set('uScale', scale);
        this._setAtlas(this.progFFT);
        this.progFFT.tex('uSrc', this.a.tex).tex('uTwiddle', this.twiddle);
        this.ctx.drawFullscreen();
        this._swap();
      }
    }
  }

  dispose() {
    this.a.fbo.dispose(); this.b.fbo.dispose();
    this.a.tex.dispose(); this.b.tex.dispose(); this.twiddle.dispose();
  }
}
