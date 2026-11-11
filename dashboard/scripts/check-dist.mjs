// Post-build checks for the GitHub Pages bundle (run by `npm run build`):
// every local asset referenced by index.html exists and sits under the configured base path,
// every dashboard data file is present with the expected protocol, and the bundle stays small.
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const dist = new URL("../dist/", import.meta.url).pathname;
const base = process.env.DASHBOARD_BASE ?? "/engine-rul/";
const fail = (msg) => {
  console.error(`check-dist: ${msg}`);
  process.exit(1);
};

const html = readFileSync(join(dist, "index.html"), "utf8");
const refs = [...html.matchAll(/(?:src|href)="([^"#]+)"/g)].map((m) => m[1]);
for (const ref of refs) {
  if (/^(https?:|data:|mailto:)/.test(ref)) continue;
  if (!ref.startsWith(base)) fail(`${ref} is not under the base path ${base}`);
  const file = join(dist, ref.slice(base.length));
  if (!existsSync(file)) fail(`${ref} is referenced but ${file} does not exist`);
}
if (!refs.some((r) => r.endsWith(".js"))) fail("no script bundle referenced from index.html");

const protocols = {
  "meta.json": "meta",
  "official_test.json": "official_test",
  "phase1_validation.json": "phase1_validation",
  "cv.json": "nested_cv",
  "ablation.json": "nested_cv_ablation",
  "conformal.json": "nested_cv_conformal",
  "benchmarks.json": "benchmark",
};
for (const [file, protocol] of Object.entries(protocols)) {
  const p = join(dist, "data", file);
  if (!existsSync(p)) fail(`missing data/${file}; run \`engine-rul export-dashboard\` first`);
  const got = JSON.parse(readFileSync(p, "utf8")).protocol;
  if (got !== protocol) fail(`data/${file} has protocol ${got}, expected ${protocol}`);
}

const size = (dir) =>
  readdirSync(dir).reduce((n, f) => {
    const p = join(dir, f);
    return n + (statSync(p).isDirectory() ? size(p) : statSync(p).size);
  }, 0);
const kb = size(dist) / 1024;
if (kb > 1500) fail(`bundle is ${kb.toFixed(0)} KB (> 1500 KB budget)`);
console.log(`check-dist: ok (${refs.length} local references, ${Object.keys(protocols).length} data files, ${kb.toFixed(0)} KB total, base ${base})`);
