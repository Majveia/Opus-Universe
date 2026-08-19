// A 3D grid of size N^3 has to live in a 2D texture, because WebGL2 cannot
// render to a 3D texture in one pass. We tile the N z-slices into a 2D atlas.
// Particles use the identical layout: particle (i,j,k) starts life in grid cell
// (i,j,k), so Lagrangian index and grid index are the same thing.

export function atlasLayout(n) {
  if ((n & (n - 1)) !== 0) throw new Error(`grid size must be a power of two, got ${n}`);
  // Tiles across is the power of two nearest sqrt(n), keeping the atlas roughly
  // square so it stays inside MAX_TEXTURE_SIZE for large grids.
  const tilesX = 1 << Math.ceil(Math.log2(Math.sqrt(n)));
  const tilesY = n / tilesX;
  return {
    n,
    tilesX,
    tilesY,
    width: n * tilesX,
    height: n * tilesY,
    count: n * n * n,
  };
}

// grid (x,y,z) -> flat texel index in the atlas
export function atlasIndex(layout, x, y, z) {
  const tx = z % layout.tilesX;
  const ty = (z / layout.tilesX) | 0;
  const px = tx * layout.n + x;
  const py = ty * layout.n + y;
  return py * layout.width + px;
}

// Repacks a value-per-cell array from grid order into atlas order.
export function gridToAtlas(layout, src, dst, components = 1, srcStride = 1, srcOffset = 0) {
  const n = layout.n;
  for (let z = 0; z < n; z++) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const gi = x + n * (y + n * z);
        const ai = atlasIndex(layout, x, y, z);
        for (let c = 0; c < components; c++) {
          dst[ai * components + c] = src[(gi * srcStride + srcOffset) + c];
        }
      }
    }
  }
  return dst;
}

// The GLSL counterpart, shared by every shader that touches the atlas.
export const ATLAS_GLSL = `
// --- 3D grid packed into a 2D atlas -------------------------------------
uniform vec4 uAtlas;   // x: N, y: 1/N, z: tilesX, w: tilesY
uniform vec2 uAtlasSize;

// Integer cell (wrapped periodically) -> texel coordinate.
ivec2 cellToTexel(ivec3 c) {
  int N = int(uAtlas.x);
  int tX = int(uAtlas.z);
  c = ((c % N) + N) % N;                     // periodic wrap
  int tx = c.z % tX;
  int ty = c.z / tX;
  return ivec2(tx * N + c.x, ty * N + c.y);
}

// Texel coordinate -> integer cell.
ivec3 texelToCell(ivec2 t) {
  int N = int(uAtlas.x);
  int tX = int(uAtlas.z);
  int cx = t.x % N;
  int cy = t.y % N;
  int tx = t.x / N;
  int ty = t.y / N;
  return ivec3(cx, cy, ty * tX + tx);
}

// Continuous grid position (in cell units) -> atlas UV, for trilinear gathers
// done manually as two bilinear taps.
vec2 cellToUV(vec3 p) {
  float N = uAtlas.x;
  float tX = uAtlas.z;
  vec3 c = mod(p, N);
  float z0 = floor(c.z);
  float tx = mod(z0, tX);
  float ty = floor(z0 / tX);
  return (vec2(tx * N + c.x, ty * N + c.y)) / uAtlasSize;
}
`;
