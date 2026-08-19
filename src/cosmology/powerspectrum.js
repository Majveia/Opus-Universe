// The linear matter power spectrum.
//
// P(k) = A k^ns T(k)^2, with T(k) from Eisenstein & Hu (1998), ApJ 496, 605 -
// the full fitting formula including the baryon acoustic oscillations. Those
// wiggles are a real, measured feature of our universe, and because the
// initial conditions are drawn from this spectrum the simulated cosmic web
// carries the acoustic scale in its clustering.
//
// Convention: k is in h/Mpc and P(k) in (Mpc/h)^3, the standard units of
// numerical cosmology. Internally the fitting formula wants k in Mpc^-1, so
// every entry point converts.

export class PowerSpectrum {
  constructor(cosmo, opts = {}) {
    this.cosmo = cosmo;
    this.useBAO = opts.useBAO !== false;
    this._fit();
    this.norm = 1;
    this.norm = this._normalizeToSigma8();
  }

  _fit() {
    const c = this.cosmo;
    const theta = c.tCMB / 2.7;
    const om = c.omegaM * c.h * c.h;    // omega_m = Omega_m h^2
    const ob = c.omegaB * c.h * c.h;    // omega_b
    const fb = c.omegaB / c.omegaM;
    const fc = 1 - fb;

    this.theta = theta; this.om = om; this.ob = ob; this.fb = fb; this.fc = fc;

    // Matter-radiation equality (EH98 eq. 1-3).
    this.zEq = 2.50e4 * om / Math.pow(theta, 4);
    this.kEq = 7.46e-2 * om / (theta * theta);          // Mpc^-1

    // Drag epoch (eq. 4).
    const b1 = 0.313 * Math.pow(om, -0.419) * (1 + 0.607 * Math.pow(om, 0.674));
    const b2 = 0.238 * Math.pow(om, 0.223);
    this.zD = 1291 * Math.pow(om, 0.251) / (1 + 0.659 * Math.pow(om, 0.828)) * (1 + b1 * Math.pow(ob, b2));

    // Baryon-photon momentum density ratio (eq. 5).
    const R = (z) => 31.5 * ob / Math.pow(theta, 4) * (1000 / z);
    this.Rd = R(this.zD);
    this.Req = R(this.zEq);

    // Sound horizon at the drag epoch (eq. 6), in Mpc.
    this.s = (2 / (3 * this.kEq)) * Math.sqrt(6 / this.Req) *
      Math.log((Math.sqrt(1 + this.Rd) + Math.sqrt(this.Rd + this.Req)) / (1 + Math.sqrt(this.Req)));

    // Silk damping scale (eq. 7), in Mpc^-1.
    this.kSilk = 1.6 * Math.pow(ob, 0.52) * Math.pow(om, 0.73) * (1 + Math.pow(10.4 * om, -0.95));

    // CDM suppression and shift (eq. 11-12).
    const a1 = Math.pow(46.9 * om, 0.670) * (1 + Math.pow(32.1 * om, -0.532));
    const a2 = Math.pow(12.0 * om, 0.424) * (1 + Math.pow(45.0 * om, -0.582));
    this.alphaC = Math.pow(a1, -fb) * Math.pow(a2, -(fb * fb * fb));
    const bb1 = 0.944 / (1 + Math.pow(458 * om, -0.708));
    const bb2 = Math.pow(0.395 * om, -0.0266);
    this.betaC = 1 / (1 + bb1 * (Math.pow(fc, bb2) - 1));

    // Baryon envelope (eq. 14-15, 23-24).
    const y = (1 + this.zEq) / (1 + this.zD);
    const sq = Math.sqrt(1 + y);
    const G = y * (-6 * sq + (2 + 3 * y) * Math.log((sq + 1) / (sq - 1)));
    this.alphaB = 2.07 * this.kEq * this.s * Math.pow(1 + this.Rd, -0.75) * G;
    this.betaB = 0.5 + fb + (3 - 2 * fb) * Math.sqrt(Math.pow(17.2 * om, 2) + 1);
    this.betaNode = 8.41 * Math.pow(om, 0.435);

    // Zero-baryon shape parameter for the smooth ("no-wiggle") variant.
    this.alphaGamma = 1 - 0.328 * Math.log(431 * om) * fb + 0.38 * Math.log(22.3 * om) * fb * fb;
    this.sApprox = 44.5 * Math.log(9.83 / om) / Math.sqrt(1 + 10 * Math.pow(ob, 0.75));
  }

  // EH98 eq. 19-20: the interpolating form used by both CDM and baryon pieces.
  _T0(q, alpha, beta) {
    const C = 14.2 / alpha + 386 / (1 + 69.9 * Math.pow(q, 1.08));
    const L = Math.log(Math.E + 1.8 * beta * q);
    return L / (L + C * q * q);
  }

  // Full transfer function with acoustic oscillations. k in Mpc^-1.
  transferMpc(k) {
    if (k <= 0) return 1;
    const q = k / (13.41 * this.kEq);
    const ks = k * this.s;

    // Cold dark matter component (eq. 17-18).
    const f = 1 / (1 + Math.pow(ks / 5.4, 4));
    const Tc = f * this._T0(q, 1, this.betaC) + (1 - f) * this._T0(q, this.alphaC, this.betaC);

    // Baryon component (eq. 21-22): a damped, node-shifted acoustic wave.
    const sTilde = this.s / Math.pow(1 + Math.pow(this.betaNode / ks, 3), 1 / 3);
    const kst = k * sTilde;
    const j0 = kst < 1e-8 ? 1 : Math.sin(kst) / kst;
    const Tb = (this._T0(q, 1, 1) / (1 + Math.pow(ks / 5.2, 2)) +
      this.alphaB / (1 + Math.pow(this.betaB / ks, 3)) * Math.exp(-Math.pow(k / this.kSilk, 1.4))) * j0;

    return this.fb * Tb + this.fc * Tc;
  }

  // Smooth, oscillation-free transfer function (EH98 sec. 4.2). Useful for
  // isolating the BAO feature by ratio.
  transferNoWiggleMpc(k) {
    if (k <= 0) return 1;
    const c = this.cosmo;
    const gammaEff = c.omegaM * c.h * (this.alphaGamma +
      (1 - this.alphaGamma) / (1 + Math.pow(0.43 * k * (this.sApprox / c.h) * c.h, 4)));
    const q = k / c.h * this.theta * this.theta / gammaEff;
    const L = Math.log(2 * Math.E + 1.8 * q);
    const C = 14.2 + 731 / (1 + 62.5 * q);
    return L / (L + C * q * q);
  }

  transfer(kh) { return this.useBAO ? this.transferMpc(kh * this.cosmo.h) : this.transferNoWiggleMpc(kh * this.cosmo.h); }
  transferNoWiggle(kh) { return this.transferNoWiggleMpc(kh * this.cosmo.h); }

  // Linear power spectrum at z = 0, in (Mpc/h)^3, for k in h/Mpc.
  P(kh) {
    if (kh <= 0) return 0;
    const T = this.transfer(kh);
    return this.norm * Math.pow(kh, this.cosmo.ns) * T * T;
  }

  // Power spectrum linearly evolved to scale factor a.
  Pa(kh, a) {
    const D = this.cosmo.growth(a);
    return this.P(kh) * D * D;
  }

  // Dimensionless power Delta^2(k) = k^3 P(k) / (2 pi^2).
  delta2(kh) { return kh * kh * kh * this.P(kh) / (2 * Math.PI * Math.PI); }

  /* ------------------------------------------------------------ moments -- */

  // Top-hat window in Fourier space.
  static windowTopHat(x) {
    if (x < 1e-4) return 1 - x * x / 10;
    return 3 * (Math.sin(x) - x * Math.cos(x)) / (x * x * x);
  }

  // Gaussian window, used when comparing against grid-smoothed fields.
  static windowGaussian(x) { return Math.exp(-0.5 * x * x); }

  // sigma^2(R) = (1/2pi^2) Int P(k) W^2(kR) k^2 dk, integrated in log k.
  sigmaSq(R, window = PowerSpectrum.windowTopHat) {
    const lnKMin = Math.log(1e-5), lnKMax = Math.log(1e3);
    const n = 4096;
    const dlnk = (lnKMax - lnKMin) / n;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const k = Math.exp(lnKMin + i * dlnk);
      const w = window(k * R);
      const f = this.P(k) * w * w * k * k * k;   // extra k from dk = k dlnk
      const wt = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2);
      sum += wt * f;
    }
    return sum * dlnk / 3 / (2 * Math.PI * Math.PI);
  }

  sigma(R, window) { return Math.sqrt(this.sigmaSq(R, window)); }

  _normalizeToSigma8() {
    const s8sq = this.sigmaSq(8.0);
    return this.cosmo.sigma8 * this.cosmo.sigma8 / s8sq;
  }

  // Mass inside a top-hat of comoving radius R, in Msun/h.
  massOfRadius(R) { return (4 / 3) * Math.PI * R * R * R * this.cosmo.rhoMeanComoving; }
  radiusOfMass(M) { return Math.pow(M / ((4 / 3) * Math.PI * this.cosmo.rhoMeanComoving), 1 / 3); }

  // Press-Schechter / Sheth-Tormen halo multiplicity, giving the comoving
  // number density of haloes per ln M. Used to populate the universe with
  // galaxies at a statistically correct abundance.
  massFunction(M, a, { A = 0.3222, p = 0.3, q = 0.707 } = {}) {
    const R = this.radiusOfMass(M);
    const sigma = this.sigma(R) * this.cosmo.growth(a);
    const nu = this.cosmo.deltaC(a) / sigma;
    const nu2 = q * nu * nu;
    const fNu = A * Math.sqrt(2 * nu2 / Math.PI) * (1 + Math.pow(nu2, -p)) * Math.exp(-nu2 / 2);
    // dln sigma^-1 / dlnM via finite difference.
    const dlnM = 0.01;
    const s1 = this.sigma(this.radiusOfMass(M * Math.exp(-dlnM))) * this.cosmo.growth(a);
    const s2 = this.sigma(this.radiusOfMass(M * Math.exp(dlnM))) * this.cosmo.growth(a);
    const dlnSigmaInvDlnM = -(Math.log(s2) - Math.log(s1)) / (2 * dlnM);
    return this.cosmo.rhoMeanComoving / M * fNu * dlnSigmaInvDlnM;
  }
}
