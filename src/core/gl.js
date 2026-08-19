// A thin, explicit WebGL2 layer. No scene graph, no material system - just the
// resources this simulation actually needs, with clear failure messages.

export class GLContext {
  constructor(canvas, opts = {}) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,          // we resolve with FXAA in the HDR pipeline
      depth: false,              // depth lives in our own framebuffers
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: !!opts.preserveDrawingBuffer,
      desynchronized: false,
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.canvas = canvas;

    this.ext = {
      colorBufferFloat: gl.getExtension('EXT_color_buffer_float'),
      floatBlend: gl.getExtension('EXT_float_blend'),
      linearFloat: gl.getExtension('OES_texture_float_linear'),
      anisotropic: gl.getExtension('EXT_texture_filter_anisotropic'),
      timerQuery: gl.getExtension('EXT_disjoint_timer_query_webgl2'),
    };
    if (!this.ext.colorBufferFloat) {
      throw new Error('EXT_color_buffer_float is required (floating point render targets).');
    }

    this.limits = {
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxVertexTextures: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
      maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
      maxSamples: gl.getParameter(gl.MAX_SAMPLES),
    };
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    this.rendererName = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

    this._quadVAO = null;
    this._emptyVAO = null;
    this.drawCalls = 0;
  }

  /* ------------------------------------------------------------ shaders -- */

  compile(type, source, label) {
    const gl = this.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error(`Shader compile failed [${label}]\n${log}\n${numberLines(source, log)}`);
    }
    return sh;
  }

  program(vsSource, fsSource, label = 'program', transformFeedbackVaryings = null) {
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, vsSource, label + ':vs');
    const fs = this.compile(gl.FRAGMENT_SHADER, fsSource, label + ':fs');
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    if (transformFeedbackVaryings) {
      gl.transformFeedbackVaryings(p, transformFeedbackVaryings, gl.SEPARATE_ATTRIBS);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      throw new Error(`Program link failed [${label}]\n${log}`);
    }
    gl.deleteShader(vs); gl.deleteShader(fs);
    return new Program(this, p, label);
  }

  /* ----------------------------------------------------------- textures -- */

  texture(opts) {
    return new Texture(this, opts);
  }

  framebuffer(textures, depth = null) {
    return new Framebuffer(this, textures, depth);
  }

  /* -------------------------------------------------------- fullscreen -- */

  // A VAO with no attributes; the vertex shader synthesises a covering triangle
  // from gl_VertexID. Avoids a vertex buffer entirely.
  get emptyVAO() {
    if (!this._emptyVAO) this._emptyVAO = this.gl.createVertexArray();
    return this._emptyVAO;
  }

  drawFullscreen() {
    const gl = this.gl;
    gl.bindVertexArray(this.emptyVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.drawCalls++;
  }

  viewport(w, h) { this.gl.viewport(0, 0, w, h); }

  bindDefaultFramebuffer(w, h) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
  }

  checkError(where = '') {
    const e = this.gl.getError();
    if (e !== this.gl.NO_ERROR) {
      const names = { 1280: 'INVALID_ENUM', 1281: 'INVALID_VALUE', 1282: 'INVALID_OPERATION', 1285: 'OUT_OF_MEMORY', 1286: 'INVALID_FRAMEBUFFER_OPERATION' };
      throw new Error(`GL error ${names[e] || e} at ${where}`);
    }
  }
}

/* ---------------------------------------------------------------- program -- */

export class Program {
  constructor(ctx, handle, label) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.handle = handle;
    this.label = label;
    this.uniforms = new Map();
    this.blocks = new Map();
    const gl = this.gl;
    const n = gl.getProgramParameter(handle, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(handle, i);
      const name = info.name.replace(/\[0\]$/, '');
      const loc = gl.getUniformLocation(handle, info.name);
      if (loc) this.uniforms.set(name, { loc, type: info.type, size: info.size });
    }
    this._unit = 0;
  }

  use() {
    this.gl.useProgram(this.handle);
    this._unit = 0;
    return this;
  }

  // Sets a uniform by name; silently ignores names the compiler optimised away
  // so call sites stay declarative.
  set(name, value) {
    const u = this.uniforms.get(name);
    if (!u) return this;
    const gl = this.gl;
    const { loc, type } = u;
    switch (type) {
      case gl.FLOAT: gl.uniform1f(loc, value); break;
      case gl.FLOAT_VEC2: gl.uniform2fv(loc, value); break;
      case gl.FLOAT_VEC3: gl.uniform3fv(loc, value); break;
      case gl.FLOAT_VEC4: gl.uniform4fv(loc, value); break;
      case gl.INT: case gl.BOOL: gl.uniform1i(loc, value); break;
      case gl.INT_VEC2: case gl.BOOL_VEC2: gl.uniform2iv(loc, value); break;
      case gl.INT_VEC3: case gl.BOOL_VEC3: gl.uniform3iv(loc, value); break;
      case gl.INT_VEC4: case gl.BOOL_VEC4: gl.uniform4iv(loc, value); break;
      case gl.UNSIGNED_INT: gl.uniform1ui(loc, value); break;
      case gl.FLOAT_MAT3: gl.uniformMatrix3fv(loc, false, value); break;
      case gl.FLOAT_MAT4: gl.uniformMatrix4fv(loc, false, value); break;
      case gl.SAMPLER_2D: case gl.SAMPLER_3D: case gl.SAMPLER_CUBE:
      case gl.SAMPLER_2D_ARRAY: case gl.INT_SAMPLER_2D: case gl.UNSIGNED_INT_SAMPLER_2D:
        gl.uniform1i(loc, value); break;
      default:
        if (typeof value === 'number') gl.uniform1f(loc, value);
        else gl.uniform4fv(loc, value);
    }
    return this;
  }

  setAll(obj) {
    for (const k in obj) this.set(k, obj[k]);
    return this;
  }

  // Binds a texture to the next free unit and points the sampler at it.
  tex(name, texture) {
    if (!this.uniforms.has(name)) return this;
    const unit = this._unit++;
    texture.bind(unit);
    this.set(name, unit);
    return this;
  }
}

/* ---------------------------------------------------------------- texture -- */

const FORMAT_TABLE = {
  // internalFormat: [format, type, channels]
  RGBA32F: ['RGBA', 'FLOAT', 4],
  RGBA16F: ['RGBA', 'HALF_FLOAT', 4],
  RG32F: ['RG', 'FLOAT', 2],
  RG16F: ['RG', 'HALF_FLOAT', 2],
  R32F: ['RED', 'FLOAT', 1],
  R16F: ['RED', 'HALF_FLOAT', 1],
  RGBA8: ['RGBA', 'UNSIGNED_BYTE', 4],
  R8: ['RED', 'UNSIGNED_BYTE', 1],
  RGBA32UI: ['RGBA_INTEGER', 'UNSIGNED_INT', 4],
  R32UI: ['RED_INTEGER', 'UNSIGNED_INT', 1],
  DEPTH_COMPONENT32F: ['DEPTH_COMPONENT', 'FLOAT', 1],
  DEPTH_COMPONENT24: ['DEPTH_COMPONENT', 'UNSIGNED_INT', 1],
};

export class Texture {
  constructor(ctx, opts) {
    const gl = ctx.gl;
    this.ctx = ctx;
    this.gl = gl;
    this.width = opts.width | 0;
    this.height = opts.height | 0;
    this.depth = opts.depth | 0;
    this.internalFormat = opts.format || 'RGBA16F';
    this.target = this.depth > 0 ? gl.TEXTURE_3D : (opts.target || gl.TEXTURE_2D);
    const entry = FORMAT_TABLE[this.internalFormat];
    if (!entry) throw new Error(`Unknown texture format ${this.internalFormat}`);
    this.format = gl[entry[0]];
    this.type = gl[entry[1]];
    this.channels = entry[2];

    this.handle = gl.createTexture();
    gl.bindTexture(this.target, this.handle);

    const filter = opts.filter === undefined ? gl.NEAREST : opts.filter;
    const wrap = opts.wrap === undefined ? gl.CLAMP_TO_EDGE : opts.wrap;
    const levels = opts.mipmap ? Math.floor(Math.log2(Math.max(this.width, this.height))) + 1 : 1;
    this.levels = levels;

    if (this.target === gl.TEXTURE_3D) {
      gl.texStorage3D(this.target, levels, gl[this.internalFormat], this.width, this.height, this.depth);
      gl.texParameteri(this.target, gl.TEXTURE_WRAP_R, wrap);
    } else {
      gl.texStorage2D(this.target, levels, gl[this.internalFormat], this.width, this.height);
    }
    gl.texParameteri(this.target, gl.TEXTURE_MIN_FILTER, opts.mipmap ? (filter === gl.LINEAR ? gl.LINEAR_MIPMAP_LINEAR : gl.NEAREST_MIPMAP_NEAREST) : filter);
    gl.texParameteri(this.target, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(this.target, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(this.target, gl.TEXTURE_WRAP_T, wrap);
    if (opts.data) this.upload(opts.data);
  }

  upload(data, level = 0) {
    const gl = this.gl;
    gl.bindTexture(this.target, this.handle);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (this.target === gl.TEXTURE_3D) {
      gl.texSubImage3D(this.target, level, 0, 0, 0, this.width, this.height, this.depth, this.format, this.type, data);
    } else {
      gl.texSubImage2D(this.target, level, 0, 0, this.width, this.height, this.format, this.type, data);
    }
    return this;
  }

  generateMipmap() {
    const gl = this.gl;
    gl.bindTexture(this.target, this.handle);
    gl.generateMipmap(this.target);
    return this;
  }

  bind(unit) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(this.target, this.handle);
    return this;
  }

  dispose() { this.gl.deleteTexture(this.handle); this.handle = null; }
}

/* ------------------------------------------------------------ framebuffer -- */

export class Framebuffer {
  constructor(ctx, textures, depth = null) {
    const gl = ctx.gl;
    this.ctx = ctx;
    this.gl = gl;
    this.textures = Array.isArray(textures) ? textures : [textures];
    this.depth = depth;
    this.width = this.textures[0].width;
    this.height = this.textures[0].height;
    this.handle = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.handle);
    const bufs = [];
    this.textures.forEach((t, i) => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t.handle, 0);
      bufs.push(gl.COLOR_ATTACHMENT0 + i);
    });
    gl.drawBuffers(bufs);
    if (depth) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth.handle, 0);
    }
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete: 0x${status.toString(16)} (${this.width}x${this.height} ${this.textures[0].internalFormat})`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  bind(setViewport = true) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.handle);
    if (setViewport) gl.viewport(0, 0, this.width, this.height);
    return this;
  }

  clear(r = 0, g = 0, b = 0, a = 0) {
    const gl = this.gl;
    this.bind();
    gl.clearColor(r, g, b, a);
    gl.clear(gl.COLOR_BUFFER_BIT | (this.depth ? gl.DEPTH_BUFFER_BIT : 0));
    return this;
  }

  readPixels(out, x = 0, y = 0, w = this.width, h = this.height, attachment = 0) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.handle);
    gl.readBuffer(gl.COLOR_ATTACHMENT0 + attachment);
    const t = this.textures[attachment];
    gl.readPixels(x, y, w, h, t.format, t.type, out);
    return out;
  }

  dispose() { this.gl.deleteFramebuffer(this.handle); }
}

/* --------------------------------------------------------------- pingpong -- */

// Two identically-configured render targets that alternate roles each step.
export class PingPong {
  constructor(ctx, opts) {
    this.a = { tex: ctx.texture(opts) };
    this.b = { tex: ctx.texture(opts) };
    this.a.fbo = ctx.framebuffer(this.a.tex);
    this.b.fbo = ctx.framebuffer(this.b.tex);
  }
  get src() { return this.a.tex; }
  get dstFBO() { return this.b.fbo; }
  get dst() { return this.b.tex; }
  swap() { const t = this.a; this.a = this.b; this.b = t; }
  dispose() {
    this.a.fbo.dispose(); this.b.fbo.dispose();
    this.a.tex.dispose(); this.b.tex.dispose();
  }
}

/* ----------------------------------------------------------------- utils -- */

export function numberLines(src, log) {
  // Surface only the neighbourhood of the first reported error line.
  const m = /ERROR:\s*\d+:(\d+)/.exec(log || '');
  const lines = src.split('\n');
  if (!m) return lines.slice(0, 40).map((l, i) => `${i + 1}: ${l}`).join('\n');
  const n = parseInt(m[1], 10);
  const lo = Math.max(0, n - 8), hi = Math.min(lines.length, n + 6);
  return lines.slice(lo, hi).map((l, i) => `${lo + i + 1}${lo + i + 1 === n ? ' >>' : ':  '} ${l}`).join('\n');
}

// The covering-triangle vertex shader used by every full-screen pass.
export const FULLSCREEN_VS = `#version 300 es
precision highp float;
out vec2 vUV;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUV = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
