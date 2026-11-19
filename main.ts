// GTA VI trophy rarity forecaster.
//
// For every trophy of Grand Theft Auto VI (PS5) this program forecasts the
// Exophase earn rate at a case's target, as five quantiles (0.05, 0.25, 0.5,
// 0.75, 0.95) of the rate in percent.
//
// Model, in one paragraph: an earn rate is a ratio, earners over players
// tracked, and Exophase refreshes the two on its own schedule, so the two
// are forecast separately and divided at the end. For the players tracked
// and for each trophy's earners I fit a weighted least-squares line of the
// log count against log time since release (log time because the launch
// surge bends over), with recent reads weighted more, and read each line at
// the target. A trophy's slope is shrunk toward the pooled slope across
// trophies, so a rare trophy with a flat history borrows the common trend
// instead of trusting its own noise. The spread is honest about three
// things: how far each trophy's own history strays from its line, a random
// walk that widens with the horizon, and the jump a refresh makes (measured
// from the history's own refreshes, with a prior when the history has seen
// none yet). Quantiles are the median's logit shifted by the standard
// normal's quantiles times that spread.
//
// Deterministic: a pure function of the case file. No network, no clock,
// no randomness, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Standard normal quantiles at the five levels the case asks for.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

const HOUR_MS = 3600000;
const DAY_MS = 86400000;

// A trophy as a case lists it.
type Trophy = {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
};

// One read of the trophy page.
type PageRead = {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
};

// One case file, schema makefun/gta6-trophies-case/v1.
type CaseFile = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: PageRead[];
};

type Fit = { a: number; b: number; seB: number; resid: number; n: number };

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

// Logit of a rate given as a fraction in (0, 1), clamped away from the ends.
function logitFrac(p: number): number {
  const q = clamp(p, 0.0001, 0.9999);
  return Math.log(q / (1 - q));
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function sdOf(values: number[]): number {
  if (values.length < 2) return 0;
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  const v = values.reduce((a, b) => a + (b - m) * (b - m), 0) / values.length;
  return Math.sqrt(v);
}

// Weighted least squares of y on x. The residual spread is floored at 0.1 so
// a series that happens to sit perfectly flat does not claim zero noise and
// a zero standard error, which would freeze its slope at exactly zero.
function wls(xs: number[], ys: number[], ws: number[]): Fit {
  const n = xs.length;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw <= 0 || n === 0) return { a: 0, b: 0, seB: 1e6, resid: 0.1, n: 0 };
  const xb = sx / sw;
  const yb = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xb;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - yb);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = yb - b * xb;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += ws[i] * r * r;
  }
  const resid = Math.max(Math.sqrt(ss / sw), 0.1);
  const seB = sxx > 1e-12 ? Math.sqrt(Math.max(ss / sw, 0.01) / sxx) : 1e6;
  return { a, b, seB, resid, n };
}

// Log hours from the release to a moment, for the trend axis.
function logHours(ms: number, releaseMs: number): number {
  const h = (ms - releaseMs) / HOUR_MS;
  return Math.log(1 + Math.max(-0.9, h));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

// Forecast one case: five quantiles of every trophy's earn rate at target.
function forecastOne(c: CaseFile): { trophies: { id: number; q: number[] }[] } {
  const trophies = c.trophies ?? [];
  const history = (c.history ?? [])
    .slice()
    .sort((p, q) => Date.parse(p.slot) - Date.parse(q.slot));
  const releaseMs = Date.parse((c.game?.release_date ?? "2026-11-19") + "T00:00:00Z");
  const targetMs = Date.parse(c.target);
  const xTarget = logHours(targetMs, releaseMs);
  const lastMs = history.length > 0 ? Date.parse(history[history.length - 1].slot) : targetMs;
  // The walk widens from the last read the case holds, not from the
  // checkpoint: the case's last read can sit before its checkpoint.
  const walkDays = Math.max(0, (targetMs - lastMs) / DAY_MS);

  // Trend axis and recency weights (half-life one day).
  const xs: number[] = [];
  const ws: number[] = [];
  for (const r of history) {
    const t = Date.parse(r.slot);
    xs.push(logHours(t, releaseMs));
    ws.push(Math.pow(0.5, Math.max(0, (lastMs - t) / DAY_MS)));
  }

  // Players tracked at each read, carrying the last good value forward.
  const ns: number[] = [];
  let lastN = 0;
  for (const r of history) {
    if (typeof r.players_tracked === "number" && r.players_tracked > 0) {
      lastN = r.players_tracked;
    }
    ns.push(lastN);
  }

  // --- Forecast the players tracked: log count against log time. ---
  let nTarget = lastN > 0 ? lastN : 1;
  {
    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let j = 0; j < history.length; j++) {
      if (ns[j] > 0) {
        px.push(xs[j]);
        py.push(Math.log(ns[j]));
        pw.push(ws[j]);
      }
    }
    if (px.length >= 3) {
      const fit = wls(px, py, pw);
      nTarget = Math.exp(fit.a + fit.b * xTarget);
    }
    // Tracked players do not fall, and a wild extrapolation is capped.
    nTarget = clamp(nTarget, lastN > 0 ? lastN : 1, (lastN > 0 ? lastN : 1) * 50);
  }

  // --- How big a refresh jump is, from the history's own refreshes. ---
  // A refresh is a read where the tracked count jumps; the logit moves of
  // every trophy across it say how far one refresh throws a rate.
  let sRefresh = 0.4; // prior: no refresh seen yet in this case's history.
  {
    const jumps: number[] = [];
    for (let j = 1; j < history.length; j++) {
      if (ns[j] > 1.05 * ns[j - 1] && ns[j - 1] > 0) {
        for (let i = 0; i < trophies.length; i++) {
          const r0 = history[j - 1].rates[i];
          const r1 = history[j].rates[i];
          if (
            typeof r0 === "number" &&
            typeof r1 === "number" &&
            r0 > 0 &&
            r0 < 100 &&
            r1 > 0 &&
            r1 < 100
          ) {
            jumps.push(logitFrac(r1 / 100) - logitFrac(r0 / 100));
          }
        }
      }
    }
    if (jumps.length >= 5) sRefresh = clamp(sdOf(jumps), 0.1, 1.0);
  }

  // --- Per-trophy earners: log count against log time, slope shrunk. ---
  const slopes: number[] = [];
  const ses: number[] = [];
  const intercepts: number[] = [];
  const hasFit: boolean[] = [];
  for (let i = 0; i < trophies.length; i++) {
    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let j = 0; j < history.length; j++) {
      const rate = history[j].rates[i];
      if (typeof rate === "number" && ns[j] > 0) {
        px.push(xs[j]);
        // Half an earner keeps a trophy nobody has yet on the log scale.
        py.push(Math.log((rate * ns[j]) / 100 + 0.5));
        pw.push(ws[j]);
      }
    }
    if (px.length >= 2) {
      const fit = wls(px, py, pw);
      slopes.push(fit.b);
      ses.push(fit.seB);
      intercepts.push(fit.a);
      hasFit.push(true);
    } else {
      slopes.push(0);
      ses.push(1e6);
      intercepts.push(px.length === 1 ? py[0] : Math.log(0.5));
      hasFit.push(false);
    }
  }
  const pooledSlope = medianOf(slopes);
  const tau = Math.max(sdOf(slopes), 0.05);
  const shrunkSlope: number[] = slopes.map((b, i) => {
    if (!hasFit[i]) return pooledSlope;
    const se2 = ses[i] * ses[i];
    const t2 = tau * tau;
    return (b / se2 + pooledSlope / t2) / (1 / se2 + 1 / t2);
  });

  // --- Per-trophy residual spread of the logit rate about its own line. ---
  const residSds: number[] = [];
  for (let i = 0; i < trophies.length; i++) {
    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let j = 0; j < history.length; j++) {
      const rate = history[j].rates[i];
      if (typeof rate === "number" && rate > 0 && rate < 100) {
        px.push(xs[j]);
        py.push(logitFrac(rate / 100));
        pw.push(ws[j]);
      }
    }
    if (px.length >= 4) {
      residSds.push(wls(px, py, pw).resid);
    }
  }
  const pooledResid = residSds.length > 0 ? medianOf(residSds) : 0.1;

  // --- Assemble each trophy's row. ---
  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < trophies.length; i++) {
    const logE = intercepts[i] + shrunkSlope[i] * xTarget;
    const earners = Math.max(Math.exp(logE) - 0.5, 0);
    const medianPct = clamp((100 * earners) / nTarget, 0.01, 99.99);
    const centre = logitFrac(medianPct / 100);

    const px: number[] = [];
    const py: number[] = [];
    const pw: number[] = [];
    for (let j = 0; j < history.length; j++) {
      const rate = history[j].rates[i];
      if (typeof rate === "number" && rate > 0 && rate < 100) {
        px.push(xs[j]);
        py.push(logitFrac(rate / 100));
        pw.push(ws[j]);
      }
    }
    const sigmaFit = px.length >= 4 ? wls(px, py, pw).resid : pooledResid;
    const spread = clamp(
      Math.sqrt(
        sigmaFit * sigmaFit +
          0.35 * 0.35 * walkDays +
          sRefresh * sRefresh,
      ),
      0.1,
      2.5,
    );

    const q: number[] = [];
    for (const z of Z) {
      const frac = 1 / (1 + Math.exp(-(centre + z * spread)));
      q.push(round4(clamp(frac * 100, 0, 100)));
    }
    // Rounding must not break the ordering.
    for (let k = 1; k < q.length; k++) {
      if (q[k] < q[k - 1]) q[k] = q[k - 1];
    }
    rows.push({ id: trophies[i].id, q });
  }
  return { trophies: rows };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const text = readFileSync(join(inDir, f), "utf8");
    const c = JSON.parse(text) as CaseFile;
    const out = forecastOne(c);
    writeFileSync(join(outDir, f.slice(0, -3) + ".out"), JSON.stringify(out));
  }
}

main();
