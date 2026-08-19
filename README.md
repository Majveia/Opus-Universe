# Opus Universe

A universe you can fly through, simulated rather than drawn.

It begins as a nearly featureless field of density fluctuations at redshift 49
and evolves forward under gravity in an expanding spacetime. Filaments thicken,
voids drain, clusters assemble at the intersections. Haloes are identified in
the particle distribution as they form, galaxies are placed in them with
properties inherited from those haloes, and any one of those galaxies can be
entered — down to a star system with planets on real Keplerian orbits.

Everything runs in the browser, in real time, with no dependencies at all: no
engine, no maths library, no shader framework. Around 7,000 lines of JavaScript
and GLSL against a raw WebGL2 context.

```
python3 -m http.server 8080     # then open http://localhost:8080
```

Or open `dist/opus-universe.html`, a single self-contained file, directly.

---

## What is actually being computed

### The expanding background

A ΛCDM cosmology with radiation, matter, curvature and a cosmological constant.
The expansion rate, the linear growth factor, cosmic time and comoving distances
are all computed from the parameters rather than tabulated:

- `E(a)` and the exact ΛCDM growth factor `D(a) = (5Ωm/2) E(a) ∫₀ᵃ da′/(a′E(a′))³`
- Cosmic time by direct integration; comoving distance by quadrature
- Collapse threshold `δc(a)` (Nakamura & Suto 1997) and virial overdensity
  `Δvir(a)` (Bryan & Norman 1998)

Verified against published values: age 13.786 Gyr, `D(z=1) = 0.608`,
`f(a) ≈ Ωm(a)^0.55`, and an Einstein–de Sitter universe reduces exactly to
`D = a` with age `2/3H₀`.

### The initial conditions

The primordial power spectrum uses the Eisenstein & Hu (1998) transfer function
**including the baryon acoustic oscillations** — so the simulated universe
carries the acoustic scale imprinted at recombination in its clustering, not
just a smooth power law.

A Gaussian random field is drawn from that spectrum and particles are displaced
off a uniform lattice using **second-order Lagrangian perturbation theory**,
which suppresses the transients that a first-order (Zel'dovich) start would
otherwise inject into the first e-folds of growth.

The generator is validated by closing the loop: draw a field from `P(k)`, then
measure `P(k)` back out of it. The mode-weighted mean ratio is 1.007, and every
bin falls within the cosmic variance expected from its mode count.

### Gravity

A **Particle-Mesh N-body solver running entirely on the GPU**. Each step:

1. Particle mass is assigned to the grid with cloud-in-cell weighting, drawn as
   eight instanced points per particle with additive float blending.
2. The density contrast is Fourier-transformed by a Stockham radix-2 FFT
   implemented as a gather, so each output texel computes its own butterfly.
   Twiddle factors are tabulated in double precision on the CPU rather than
   evaluated with `sin`/`cos` in the shader — that alone was costing three
   orders of magnitude of accuracy on drivers with fast transcendentals.
3. Poisson's equation is solved by dividing by `−k²`, with the cloud-in-cell
   assignment window deconvolved.
4. The force is obtained by **exact spectral differentiation** (`F̂ = −i2πn ψ̂`),
   recovering all three components from two inverse transforms by packing two
   real fields into one complex one. A finite difference loses a third of the
   force at the mesh scale; this loses none.
5. Positions and momenta are advanced by a symplectic leapfrog in the scale
   factor, with momenta staggered half a step behind positions.

The GPU FFT is checked against a CPU reference to float32 precision
(3.4 × 10⁻⁷ relative error). The solver is checked against linear theory:

| k [h/Mpc] | amplitude growth ÷ linear | correlation with initial field |
|-----------|---------------------------|-------------------------------|
| 0.045     | 0.95                      | 0.995                         |
| 0.085     | 0.88                      | 0.975                         |
| 0.133     | 0.93                      | 0.867                         |
| 0.210     | 0.94                      | 0.573                         |
| 0.337     | **1.19**                  | 0.211                         |

Large scales track the ΛCDM growth factor; small scales overshoot it because
they have gone nonlinear, and their correlation with the initial conditions
falls away as modes couple. Over a full run the box acquires no net momentum
(|⟨p⟩|/p_rms ≈ 10⁻⁹), and the density field goes from Gaussian to strongly
skewed — voids empty, knots collapse.

### Haloes and galaxies

Structures are found with **friends-of-friends** at the standard linking length
of 0.2 mean interparticle separations. The definition depends on no mesh, so it
stays meaningful at any resolution; what resolution changes is the smallest
halo that can be resolved, which the interface reports rather than hides.

Each halo's bulk velocity, angular momentum and internal velocity dispersion are
measured from its own member particles. Galaxies then inherit:

- **stellar mass** by abundance matching (Behroozi et al. 2013) — peaking at
  2.7% efficiency around a 10¹² M⊙ halo and falling off both ways, as feedback
  and quenching require
- **disk size** from the halo's measured spin (Mo, Mao & White 1998), which puts
  the median spiral scale radius at 1.8 kpc/h — the Milky Way's value
- **orientation** from the halo's angular momentum vector, so galaxy spins trace
  the tidal field the simulation produced
- **satellites** drawn from a subhalo mass function and distributed on an NFW
  profile

Morphology follows from mass, spin and environment, and reproduces the observed
**morphology–density relation** without being told to: 62% early-type among
cluster satellites and 95% for the central galaxy of a rich cluster, against 13%
around a Milky Way-mass host.

Galaxies are drawn by raymarching a small volume — an exponential disk with a
vertical scale height, logarithmic spiral arms, a Sérsic bulge, HII regions on
the arms, and an absorbing dust layer offset to their leading edge. Because it
is a volume and not a sprite, an edge-on disk shows a dust lane cutting across a
bright bulge, from the same parameters that produce the face-on spiral.

### Star systems

Entering a galaxy generates a star system deterministically from its position,
so the same galaxy always contains the same stars.

- The star is drawn from a **Kroupa initial mass function weighted by
  main-sequence lifetime**, which turns the birth mass function into the
  present-day population. The result is 80% M dwarfs and 0.25% B stars against
  76% and 0.13% observed — the sky really is almost all red dwarfs.
- Luminosity, radius and temperature follow main-sequence structure; colour comes
  from the Planckian locus in CIE chromaticity space, converted to linear sRGB.
- Binary fraction rises with primary mass: 27% for M dwarfs, 43% for G, 74% for A
  (observed: 27%, 44%, 60%).
- Planets are spaced by **mutual Hill stability**, with giants forming beyond a
  snow line set by the star's luminosity and requiring metallicity to do it.
  Equilibrium temperature and mass decide whether a world ends up molten, rocky,
  desert, terran, ocean, icy, or a giant. Rings, moons, obliquity and tidal
  locking follow.
- Orbits are genuine Keplerian ellipses with all six elements, advanced by
  solving Kepler's equation, so `T² ∝ a³/M` holds and inner planets lap outer
  ones.

Planets are rendered as analytic spheres — no mesh, no level of detail, perfectly
round from any distance — with **integrated Rayleigh and Mie scattering** using
physical coefficients and scale heights. The blue sky, the reddened terminator,
the bright limb and the thin rim of light on the night side all fall out of the
same integral.

The night sky inside a system is generated by sampling stellar positions from the
host galaxy's own density profile, placing the observer inside it, and working out
each star's apparent brightness, colour and dust reddening from there. The band
of the Milky Way is not drawn: it appears because looking along a disk means
looking through far more stars than looking out of it.

---

## Controls

| | |
|---|---|
| `W A S D` | fly |
| `Space` / `Shift` | up / down |
| `Q` `E` | roll |
| drag or click to lock | look |
| scroll | speed (logarithmic, spans metres to megaparsecs) |
| `F` | boost |
| `X` | descend into the galaxy or body you are facing |
| `Z` | back out one scale |
| `T` | next body in the system |
| `O` | orbit lock on the current target |
| `P` | pause cosmic time |
| `[` `]` | time rate |
| `R` | rebuild from new initial conditions |
| `1`–`4` | vantage points |
| `M` / `G` | dark matter / galaxies |
| `Tab` | parameters |
| `H` | hide the interface |

Cosmological parameters can be changed and the universe rebuilt from them. Turn
Ωm down and the web grows anaemic; turn σ8 up and clusters arrive early.
Einstein–de Sitter and Λ-dominated presets are included for comparison.

URL parameters: `?seed=`, `?grid=`, `?box=`, `?steps=`, `?maxpixels=`.

---

## Layout

```
src/core/       maths, RNG, WebGL2 layer, camera, input
src/cosmology/  ΛCDM background, Eisenstein-Hu spectrum, FFT, 2LPT
src/sim/        atlas packing, GPU FFT, Particle-Mesh solver
src/universe/   halo finding, galaxy population, star systems
src/render/     HDR chain, cosmic web, galaxies, bodies, starfield, colour
tools/build.js  dependency-free bundler -> dist/opus-universe.html
test/           numerical and rendering tests
```

## Tests

```
node test/fft.test.mjs        # FFT against a naive DFT, Parseval, Hermitian symmetry
node test/ics.test.mjs        # initial conditions: P(k) measured back out of the field
node test/gpufft.test.mjs     # GPU FFT against the CPU reference
node test/pm.test.mjs         # N-body growth against linear theory, mesh convergence
node test/galaxy.mjs          # halo finding, galaxy population, rendering
node test/visual.mjs          # loads the app, evolves it, captures screenshots
```

The rendering tests drive real headless Chromium with a software GL backend, so
they check what the GPU actually produces rather than what the code intends.

## Rendering notes

The cosmic web is drawn as **projected mass column density**: each particle
carries a smoothing length scaling as ρ^(−1/3), as in smoothed-particle
hydrodynamics, so voids read as empty rather than grainy and cluster cores stay
compact. The normalisation makes mean surface brightness independent of both
particle count and screen resolution, which is why the same exposure works at
every quality setting.

The periodic box is tiled around the camera with frustum culling and sub-lattice
decimation for distant copies. There is no edge to the universe; fly in any
direction and structure keeps coming.

Output goes through an HDR chain: energy-conserving bloom with a Karis average
on the first tap, the ACES filmic transform, and ordered dithering before the
8-bit quantise. That last step matters more than it sounds — this scene is almost
entirely near-black gradients, and without dithering they band into visible
contour rings on an OLED panel.
