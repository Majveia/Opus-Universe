// The heads-up display. Deliberately sparse: a few numbers that mean something,
// set in tabular figures so they do not jitter, and nothing that needs a border
// to be legible against a black sky.

const $ = (id) => document.getElementById(id);

// Named epochs of cosmic history, by redshift. The boundaries are real ones -
// recombination, reionisation, the peak of cosmic star formation, and the
// moment dark energy overtakes matter.
const EPOCHS = [
  { z: 1100, name: 'recombination' },
  { z: 30, name: 'the dark ages' },
  { z: 15, name: 'cosmic dawn' },
  { z: 6, name: 'reionisation' },
  { z: 3, name: 'early galaxy assembly' },
  { z: 1.5, name: 'cosmic noon' },
  { z: 0.7, name: 'clusters assembling' },
  { z: 0.3, name: 'dark energy takes over' },
  { z: 0.05, name: 'accelerating expansion' },
  { z: -1, name: 'the present day' },
];

function epochName(z) {
  for (const e of EPOCHS) if (z > e.z) return e.name;
  return 'the present day';
}

function fmtCount(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n | 0);
}

export class Hud {
  constructor() {
    this.lastTicks = null;
    this.flashEl = document.createElement('div');
    Object.assign(this.flashEl.style, {
      position: 'absolute', left: '50%', top: '58%', transform: 'translate(-50%,-50%)',
      fontSize: '10.5px', letterSpacing: '.24em', textTransform: 'uppercase',
      color: 'rgba(255,255,255,.6)', opacity: '0', transition: 'opacity 500ms ease',
      pointerEvents: 'none', fontFamily: 'var(--sans)',
    });
    $('hud').appendChild(this.flashEl);
    this._flashTimer = null;
    this._frame = 0;
  }

  flash(text) {
    this.flashEl.textContent = text;
    this.flashEl.style.opacity = '1';
    clearTimeout(this._flashTimer);
    this._flashTimer = setTimeout(() => { this.flashEl.style.opacity = '0'; }, 1500);
  }

  update({ universe, camera, web, state, fps, galaxies, stellar }) {
    // Most of this changes slowly; refreshing a third as often keeps layout
    // work out of the frame budget without any visible lag.
    this._frame++;
    if (this._frame % 3 !== 0) return;

    const pm = universe.pm;
    if (!pm) return;

    if (state.scale === 'stellar' && stellar && stellar.system) {
      this._stellar(stellar, state, camera, fps);
      return;
    }
    $('target').classList.remove('on');
    $('timeline').style.opacity = '';
    const z = universe.redshift;
    const a = universe.a;

    $('redshiftValue').innerHTML = z > 0.0005
      ? `z <small>=</small> ${z >= 10 ? z.toFixed(1) : z.toFixed(3)}`
      : 'z <small>=</small> 0';

    const age = universe.ageGyr;
    $('ageValue').textContent = `${age.toFixed(3)} Gyr after the Big Bang`;
    $('phaseValue').textContent = epochName(z);

    $('tAcale').textContent = a.toFixed(4);
    $('tParticles').textContent = fmtCount(pm.particleCount);
    $('tSplats').textContent = fmtCount(web.stats.points) + ` / ${web.stats.tiles}`;
    $('tFps').textContent = fps.toFixed(0);
    const gs = state.galaxyStats;
    $('tGalaxies').textContent = gs
      ? `${fmtCount(gs.count)}  ${gs.spirals}S ${gs.ellipticals}E`
      : (state.showGalaxies ? '—' : 'off');

    const spd = camera.speed * camera.boost;
    $('tSpeed').textContent = spd >= 1 ? `${spd.toFixed(1)} Mpc/h s⁻¹` : `${(spd * 1000).toFixed(0)} kpc/h s⁻¹`;

    // Linear structure amplitude at this epoch: sigma_8 grown by D(a).
    const sigma = universe.cosmo.sigma8 * universe.cosmo.growth(a);
    $('tSigma').textContent = sigma.toFixed(3);

    /* ------------------------------------------------------- timeline -- */
    const lnA0 = Math.log(pm.aInit), lnA1 = Math.log(pm.aMax);
    const t = (Math.log(a) - lnA0) / (lnA1 - lnA0);
    $('tlFill').style.width = `${(t * 100).toFixed(2)}%`;
    $('tlHead').style.left = `${(t * 100).toFixed(2)}%`;
    $('tlLeft').textContent = `z = ${(1 / pm.aInit - 1).toFixed(0)}`;

    if (!this.lastTicks || this._ticksScale !== 'cosmic') {
      this._ticksScale = 'cosmic';
      const ticks = [20, 10, 6, 3, 2, 1, 0.5, 0.2, 0];
      const html = ticks.map((zt) => {
        const at = 1 / (1 + zt);
        const p = (Math.log(at) - lnA0) / (lnA1 - lnA0);
        if (p < 0 || p > 1) return '';
        return `<span class="tick" style="left:${(p * 100).toFixed(2)}%">${zt === 0 ? 'now' : zt}</span>`;
      }).join('');
      $('tlTicks').innerHTML = html;
      this.lastTicks = true;
    }
  }

  // A second readout for when the camera is inside a star system. The numbers
  // are the ones an astronomer would want: orbital distance and period, mass
  // and radius against Earth, surface temperature, and whether the world sits
  // in its star's habitable zone.
  _stellar(stellar, state, camera, fps) {
    const s = stellar.system;
    const star = s.star;
    $('redshiftValue').innerHTML = s.name;
    $('ageValue').textContent = `${star.class}-type star · ${star.mass.toFixed(2)} M☉ · ${star.temperature.toFixed(0)} K`;
    $('phaseValue').textContent = `${s.planets.length} planet${s.planets.length === 1 ? '' : 's'}`
      + (s.habitableCount ? ` · ${s.habitableCount} in the habitable zone` : '')
      + (s.companion ? ' · binary' : '');

    $('tAcale').textContent = `${stellar.timeYears.toFixed(2)} yr`;
    $('tSigma').textContent = `${state.yearsPerSecond.toFixed(3)} yr/s`;
    $('tParticles').textContent = `${s.belts.reduce((a, b) => a + b.count, 0)} bodies`;
    $('tSplats').textContent = `${(this._sky || 0)} stars`;
    $('tGalaxies').textContent = `${s.rSnow.toFixed(2)} AU snow line`;
    $('tFps').textContent = fps.toFixed(0);
    const spd = camera.speed * camera.boost;
    $('tSpeed').textContent = spd >= 0.01 ? `${spd.toFixed(3)} AU s⁻¹` : `${(spd * 1.496e8).toExponential(1)} km s⁻¹`;

    const t = $('target');
    const b = stellar.placed[state.targetBodyIndex];
    if (b) {
      const p = b.ref;
      const rows = [`<div class="h">${b.label}</div>`];
      if (b.kind === 'star') {
        rows.push(row('luminosity', `${p.luminosity.toExponential(2)} L☉`));
        rows.push(row('radius', `${p.radius.toFixed(2)} R☉`));
        rows.push(row('temperature', `${p.temperature.toFixed(0)} K`));
      } else if (b.kind === 'planet') {
        rows.push(row('orbit', `${p.semiMajorAU.toFixed(3)} AU  e=${p.eccentricity.toFixed(3)}`));
        rows.push(row('period', p.periodYears < 1
          ? `${(p.periodYears * 365.25).toFixed(1)} days` : `${p.periodYears.toFixed(2)} years`));
        rows.push(row('mass', `${p.massEarth < 10 ? p.massEarth.toFixed(2) : p.massEarth.toFixed(0)} M⊕`));
        rows.push(row('radius', `${p.radiusEarth.toFixed(2)} R⊕`));
        rows.push(row('surface', `${p.tSurface.toFixed(0)} K  (${(p.tSurface - 273.15).toFixed(0)} °C)`));
        if (p.moons.length) rows.push(row('moons', String(p.moons.length)));
        if (p.hasRings) rows.push(row('rings', 'yes'));
        if (p.tidallyLocked) rows.push(row('rotation', 'tidally locked'));
        if (p.habitable) rows.push('<div class="hz">within the habitable zone</div>');
      } else {
        rows.push(row('orbit', `${p.semiMajorAU.toExponential(2)} AU`));
        rows.push(row('period', `${(p.periodYears * 365.25).toFixed(2)} days`));
        rows.push(row('radius', `${(p.radiusAU * 23454.8).toFixed(3)} R⊕`));
      }
      t.innerHTML = rows.join('');
      t.classList.add('on');
    } else {
      t.classList.remove('on');
    }

    // The cosmic timeline means nothing here.
    $('timeline').style.opacity = '0';
  }
}

function row(k, v) { return `<div><span class="k">${k}</span>${v}</div>`; }
