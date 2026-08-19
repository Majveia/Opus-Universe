// Flat-or-curved LCDM background cosmology.
//
// Everything the simulation needs to know about how the universe expands lives
// here: the expansion rate, the linear growth factor that sets how fast
// structure assembles, distances, and cosmic time.
//
// Units follow the standard convention of numerical cosmology:
//   - lengths in comoving Mpc/h
//   - H0 = 100 h km/s/Mpc, so H0 = 1 in units where time is measured in
//     Hubble times and length in Mpc/h.

export const C_LIGHT_KMS = 299792.458;          // km/s
export const HUBBLE_TIME_GYR = 9.77792221;      // (1/H0) in Gyr, for h = 1
export const HUBBLE_DIST_MPC = 2997.92458;      // (c/H0) in Mpc, for h = 1

// Planck 2018 TT,TE,EE+lowE+lensing+BAO best fit (Aghanim et al. 2020, Table 2).
export const PLANCK18 = Object.freeze({
  h: 0.6766,
  omegaM: 0.3111,
  omegaB: 0.04897,
  omegaLambda: 0.6889,
  ns: 0.9665,
  sigma8: 0.8102,
  tCMB: 2.7255,
});

// Alternative parameter sets, useful for showing how the cosmic web responds
// to the underlying physics.
export const PRESETS = Object.freeze({
  planck18: { label: 'Planck 2018', ...PLANCK18 },
  wmap9: { label: 'WMAP 9', h: 0.6932, omegaM: 0.2865, omegaB: 0.04628, omegaLambda: 0.7135, ns: 0.9608, sigma8: 0.820, tCMB: 2.7255 },
  eds: { label: 'Einstein-de Sitter', h: 0.70, omegaM: 1.0, omegaB: 0.0455, omegaLambda: 0.0, ns: 0.96, sigma8: 0.80, tCMB: 2.7255 },
  openLowDensity: { label: 'Open, low density', h: 0.70, omegaM: 0.20, omegaB: 0.04, omegaLambda: 0.0, ns: 0.96, sigma8: 0.80, tCMB: 2.7255 },
  lambdaHeavy: { label: 'Lambda dominated', h: 0.70, omegaM: 0.15, omegaB: 0.03, omegaLambda: 0.85, ns: 0.96, sigma8: 0.85, tCMB: 2.7255 },
});

export class Cosmology {
  constructor(params = {}) {
    const p = { ...PLANCK18, ...params };
    this.h = p.h;
    this.omegaM = p.omegaM;
    this.omegaB = p.omegaB;
    this.omegaLambda = p.omegaLambda;
    this.ns = p.ns;
    this.sigma8 = p.sigma8;
    this.tCMB = p.tCMB;
    this.omegaK = 1 - this.omegaM - this.omegaLambda;
    // Photon + massless neutrino density; small, but it matters at z ~ 1000
    // and keeps the early-time expansion honest.
    this.omegaR = 4.165e-5 / (this.h * this.h) * Math.pow(this.tCMB / 2.7255, 4);

    this._buildGrowthTable();
    this._buildTimeTable();
  }

  /* --------------------------------------------------------- background -- */

  // E(a) = H(a)/H0.
  E(a) {
    const a2 = a * a, a3 = a2 * a, a4 = a2 * a2;
    return Math.sqrt(this.omegaR / a4 + this.omegaM / a3 + this.omegaK / a2 + this.omegaLambda);
  }

  // dE/da, used for the analytic derivative of the growth factor.
  dEda(a) {
    const a3 = a * a * a, a4 = a3 * a, a5 = a4 * a;
    return (-4 * this.omegaR / a5 - 3 * this.omegaM / a4 - 2 * this.omegaK / a3) / (2 * this.E(a));
  }

  H(a) { return 100 * this.h * this.E(a); }               // km/s/Mpc
  Ez(z) { return this.E(1 / (1 + z)); }

  // Matter density parameter at scale factor a.
  omegaMz(a) { const e = this.E(a); return this.omegaM / (a * a * a * e * e); }

  // Critical density in Msun h^2 / (Mpc/h)^3.
  get rhoCrit0() { return 2.77536627e11; }
  // Mean matter density in Msun/h per (Mpc/h)^3, constant in comoving units.
  get rhoMeanComoving() { return this.rhoCrit0 * this.omegaM; }

  /* ------------------------------------------------------------- growth -- */

  // Exact LCDM linear growth: D(a) = (5 Om / 2) E(a) Int_0^a da' / (a' E(a'))^3.
  // Reduces to D = a for Einstein-de Sitter, which the test suite checks.
  _growthIntegral(a) {
    // The integrand diverges as a'^{-1/2} at the origin for matter domination,
    // so substitute a' = u^2 to make it smooth and integrate in u.
    const n = 2048;
    const uMax = Math.sqrt(a);
    const du = uMax / n;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const u = i * du;
      const ap = u * u;
      let f;
      if (ap < 1e-12) {
        // lim_{a->0} 2u / (a E)^3 with E -> sqrt(Om/a^3) gives 2 u^4 / Om^1.5.
        f = 0;
      } else {
        const e = this.E(ap);
        f = 2 * u / Math.pow(ap * e, 3);
      }
      const w = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2);
      sum += w * f;
    }
    return sum * du / 3;
  }

  _growthRaw(a) { return 2.5 * this.omegaM * this.E(a) * this._growthIntegral(a); }

  _buildGrowthTable() {
    this._growthNorm = this._growthRaw(1);
    const N = 512;
    this._gLnAMin = Math.log(1e-4);
    this._gLnAMax = Math.log(1.0);
    // Tabulated in log-log. Both D and dD/da are power laws in a at early
    // times, so linear interpolation of the logs is exact where it matters most
    // - the high-redshift start of the simulation, where an amplitude error
    // would propagate into every structure that later forms.
    this._gTable = new Float64Array(N);
    this._gDeriv = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const a = Math.exp(this._gLnAMin + (this._gLnAMax - this._gLnAMin) * i / (N - 1));
      this._gTable[i] = Math.log(this._growthRaw(a) / this._growthNorm);
      this._gDeriv[i] = Math.log(this._dGrowthRawDa(a) / this._growthNorm);
    }
    this._gN = N;
  }

  _dGrowthRawDa(a) {
    // d/da [ (5Om/2) E I ] = (5Om/2) [ E' I + E * 1/(aE)^3 ]
    const E = this.E(a);
    return 2.5 * this.omegaM * (this.dEda(a) * this._growthIntegral(a) + 1 / (a * a * a * E * E));
  }

  // Interpolates a log-tabulated quantity and returns it in linear space.
  _tableLookup(table, a, slopeBelow) {
    const lnA = Math.log(Math.max(a, 1e-30));
    const t = (lnA - this._gLnAMin) / (this._gLnAMax - this._gLnAMin) * (this._gN - 1);
    if (t <= 0) {
      // Extrapolate along the known early-time power law rather than clamping.
      return Math.exp(table[0] + slopeBelow * (lnA - this._gLnAMin));
    }
    if (t >= this._gN - 1) return Math.exp(table[this._gN - 1]);
    const i = Math.floor(t), f = t - i;
    return Math.exp(table[i] * (1 - f) + table[i + 1] * f);
  }

  // Linear growth factor normalised so D(a = 1) = 1.
  growth(a) {
    if (a > 1) return this._growthRaw(a) / this._growthNorm;
    return this._tableLookup(this._gTable, a, 1);      // D  ~ a

  }
  growthZ(z) { return this.growth(1 / (1 + z)); }

  // dD/da.
  growthDeriv(a) {
    if (a > 1) return this._dGrowthRawDa(a) / this._growthNorm;
    return this._tableLookup(this._gDeriv, a, 0);      // dD/da ~ const

  }

  // Logarithmic growth rate f = dlnD/dlna. Approximately Omega_m(a)^0.55.
  growthRate(a) { return a * this.growthDeriv(a) / Math.max(this.growth(a), 1e-30); }

  /* --------------------------------------------------------------- time -- */

  _buildTimeTable() {
    // t(a) = Int_0^a da' / (a' H(a')); substitute a' = u^2 again for smoothness.
    const N = 512;
    this._tLnAMin = Math.log(1e-6);
    this._tLnAMax = Math.log(2.0);
    this._tTable = new Float64Array(N);
    this._tN = N;
    for (let i = 0; i < N; i++) {
      const a = Math.exp(this._tLnAMin + (this._tLnAMax - this._tLnAMin) * i / (N - 1));
      this._tTable[i] = this._ageIntegral(a);
    }
  }

  _ageIntegral(a) {
    const n = 1024;
    const uMax = Math.sqrt(a), du = uMax / n;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const u = i * du, ap = u * u;
      let f = 0;
      if (ap > 1e-14) f = 2 * u / (ap * this.E(ap));
      else {
        // a -> 0 with radiation: E -> sqrt(Or)/a^2, integrand -> 2 u^3 / sqrt(Or)
        f = 2 * u * u * u / Math.sqrt(this.omegaR);
      }
      const w = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2);
      sum += w * f;
    }
    return sum * du / 3;
  }

  // Age of the universe at scale factor a, in Gyr.
  ageGyr(a) {
    const lnA = Math.log(Math.max(a, 1e-8));
    const t = (lnA - this._tLnAMin) / (this._tLnAMax - this._tLnAMin) * (this._tN - 1);
    let val;
    if (t <= 0) val = this._ageIntegral(a);
    else if (t >= this._tN - 1) val = this._ageIntegral(a);
    else {
      const i = Math.floor(t), f = t - i;
      val = this._tTable[i] * (1 - f) + this._tTable[i + 1] * f;
    }
    return val * HUBBLE_TIME_GYR / this.h;
  }

  get ageNowGyr() { return this.ageGyr(1); }
  lookbackGyr(a) { return this.ageNowGyr - this.ageGyr(a); }

  /* ---------------------------------------------------------- distances -- */

  // Comoving distance to redshift z in Mpc/h.
  comovingDistance(z) {
    const n = 512;
    const dz = z / n;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const zi = i * dz;
      const f = 1 / this.Ez(zi);
      const w = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2);
      sum += w * f;
    }
    return sum * dz / 3 * HUBBLE_DIST_MPC;  // already in Mpc/h since c/H0 * h
  }

  luminosityDistance(z) { return this.comovingDistance(z) * (1 + z); }
  angularDiameterDistance(z) { return this.comovingDistance(z) / (1 + z); }

  // Invert comoving distance -> redshift by bisection (light-cone rendering).
  redshiftAtDistance(dMpcH) {
    let lo = 0, hi = 1100;
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi);
      if (this.comovingDistance(mid) < dMpcH) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  /* ------------------------------------------------- collapse thresholds -- */

  // Critical linear overdensity for spherical collapse. The weak cosmology
  // dependence follows Nakamura & Suto (1997).
  deltaC(a) {
    const om = this.omegaMz(a);
    return (3 / 20) * Math.pow(12 * Math.PI, 2 / 3) * (1 + 0.0123 * Math.log10(om));
  }

  // Virial overdensity relative to the mean, from the fit of Bryan & Norman (1998).
  deltaVir(a) {
    const om = this.omegaMz(a);
    const x = om - 1;
    return (18 * Math.PI * Math.PI + 82 * x - 39 * x * x) / om;
  }

  describe() {
    return {
      h: this.h, omegaM: this.omegaM, omegaB: this.omegaB, omegaLambda: this.omegaLambda,
      omegaK: this.omegaK, omegaR: this.omegaR, ns: this.ns, sigma8: this.sigma8,
      ageGyr: this.ageNowGyr,
    };
  }
}
