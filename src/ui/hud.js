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

  update({ universe, camera, web, state, fps }) {
    // Most of this changes slowly; refreshing a third as often keeps layout
    // work out of the frame budget without any visible lag.
    this._frame++;
    if (this._frame % 3 !== 0) return;

    const pm = universe.pm;
    if (!pm) return;
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

    if (!this.lastTicks) {
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
}
