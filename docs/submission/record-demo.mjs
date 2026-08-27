import { chromium } from 'file:///D:/Hackathon/node_modules/playwright/index.mjs';
const OUT = 'C:/Users/SHIVNA~1/AppData/Local/Temp/claude/D--Hackathon/d798e8fa-b624-4f63-baf1-31c876b52a18/scratchpad/final';
const BASE = 'https://continuity-609752596743.us-central1.run.app';

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1280, height: 720 },
  recordVideo: { dir: OUT, size: { width: 1280, height: 720 } },
});
const page = await ctx.newPage();

// Caption overlay injected into the page, so the recording captures it natively and the
// timing is exact -- no post-processing, and Playwright's bundled ffmpeg has no drawtext.
const installOverlay = async () => page.evaluate(() => {
  if (document.getElementById('__cap')) return;
  const band = document.createElement('div');
  band.id = '__cap';
  band.style.cssText = `position:fixed;left:0;right:0;bottom:0;z-index:99999;
    background:linear-gradient(to top,rgba(0,0,0,.92),rgba(0,0,0,.72) 70%,transparent);
    padding:22px 40px 24px;text-align:center;font:500 25px/1.35 system-ui,-apple-system,sans-serif;
    color:#fff;pointer-events:none;letter-spacing:.1px;transition:opacity .25s;`;
  document.body.appendChild(band);
  const mark = document.createElement('div');
  mark.style.cssText = `position:fixed;top:14px;right:18px;z-index:99999;color:rgba(255,255,255,.5);
    font:500 14px system-ui,sans-serif;pointer-events:none;letter-spacing:.3px;`;
  mark.textContent = 'real-time \u00b7 not sped up';
  document.body.appendChild(mark);
  window.__say = (t) => { const b = document.getElementById('__cap'); b.style.opacity = t ? '1' : '0'; b.textContent = t || ''; };
});
const say = async (t) => { await installOverlay(); await page.evaluate((x) => window.__say(x), t); };


// Full-screen title/result cards. Same injection approach as the captions -- the page
// records them natively, so no post-processing and exact timing.
const card = async (lines, ms) => {
  await page.evaluate((ls) => {
    let c = document.getElementById('__card');
    if (!c) {
      c = document.createElement('div');
      c.id = '__card';
      c.style.cssText = `position:fixed;inset:0;z-index:100000;background:#0a0a0b;
        display:flex;flex-direction:column;align-items:center;justify-content:center;
        gap:20px;padding:0 90px;text-align:center;font-family:system-ui,-apple-system,sans-serif;
        transition:opacity .4s;pointer-events:none;`;
      document.body.appendChild(c);
    }
    c.style.display = 'flex';
    c.style.opacity = '1';
    c.innerHTML = ls.map((l, i) =>
      i === 0
        ? `<div style="font:600 40px/1.25 system-ui;color:#fff;letter-spacing:-.5px">${l}</div>`
        : `<div style="font:400 25px/1.5 system-ui;color:rgba(255,255,255,.66);max-width:920px">${l}</div>`
    ).join('');
  }, lines);
  await page.waitForTimeout(ms);
};
const hideCard = async () => {
  await page.evaluate(() => { const c = document.getElementById('__card'); if (c) c.style.opacity = '0'; });
  await page.waitForTimeout(500);
  await page.evaluate(() => { const c = document.getElementById('__card'); if (c) c.style.display = 'none'; });
};

const t0 = Date.now();
const at = (l) => console.log(`  ${((Date.now()-t0)/1000).toFixed(1)}s  ${l}`);

await page.goto(BASE, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
await card(['When streaming quality breaks for some of your viewers,',
            'the dashboard shows the spike immediately. That is not the hard part.'], 5000);
await card(['Who is affected, what caused it, and what it costs',
            'takes a senior analyst three to five days — because it is not one query. ' +
            'It is a chain, and each step depends on the answer to the last.'], 6000);
await hideCard();
await say('Continuity \u2014 63.8M playback events in ClickHouse Cloud');
await page.waitForTimeout(4500); at('feed');

await say('Pick an incident. Detection is pure SQL \u2014 no model involved yet.');
await page.getByText('INC-APP-ROKU-820').first().click();
await page.getByText('no model, $0').waitFor({ timeout: 60000 });
await page.waitForTimeout(3500); at('detect');

await say('The deterministic control arm runs alongside, for comparison');
await page.getByText(/answered in/).first().waitFor({ timeout: 180000 });
await page.waitForTimeout(2500); at('control answered');

await say('Now Gemini investigates \u2014 one frame per measurement, as it happens');
let seen = 0, said2 = false;
for (let i = 0; i < 90; i++) {
  const n = await page.locator('ol li').count();
  if (n > seen) { seen = n; await page.locator('ol li').nth(n-1).scrollIntoViewIfNeeded(); }
  if (n >= 3 && !said2) { said2 = true; await say('It narrows: the population, then Roku, then app 8.2.0 \u2014 and stops'); }
  if (await page.getByText('Brief', { exact: true }).first().count() > 0) break;
  await page.waitForTimeout(1800);
}
at(`investigation done (${seen} measurements)`);

await say('Every step opens the exact ClickHouse query behind it');
const sql = page.locator('summary', { hasText: 'View the query' }).first();
if (await sql.count()) { await sql.scrollIntoViewIfNeeded(); await sql.click(); await page.waitForTimeout(5000); await sql.click(); }
at('sql');

await say('The model had to engage with evidence against its own conclusion');
const dis = page.getByText(/Disconfirming evidence/i).first();
if (await dis.count()) { await dis.scrollIntoViewIfNeeded(); await page.waitForTimeout(4500); }

await say('Every claim traced to the measurement that produced it \u2014 checked mechanically');
const cl = page.getByText(/Every claim, traced/i).first();
if (await cl.count()) { await cl.scrollIntoViewIfNeeded(); await page.waitForTimeout(4500); }
at('brief');

await say('The rollback is a proposal. Nothing acts without a human.');
const approve = page.getByRole('button', { name: 'Approve' });
if (await approve.count()) { await approve.scrollIntoViewIfNeeded(); await page.waitForTimeout(2000); await approve.click(); await page.waitForTimeout(3500); }
at('approved');

await say('');
await card(['Measured against a control arm',
            'A second implementation solves the same problem with pure statistics and no AI. ' +
            'Both run the same planted incidents, scored by the same code.'], 5600);
await card(['Agent 3/3 &nbsp;&middot;&nbsp; Walker 2/3',
            'exact blast radius — zero errors in either arm',
            '<span style="font-size:21px">On the incident it wins, a per-title encode fault, the walker lands on ' +
            'cdn/pop/app_version and understates revenue at risk by 87% ' +
            '($212 against $1,659).</span>'], 7000);
await card(['And it says when it cannot explain something',
            'On that same incident the agent could not corroborate a cause, so it marked its own ' +
            'estimate unreliable rather than blaming the nearest plausible change. It scored zero ' +
            'for attribution — the same as a confidently wrong answer.'], 7000);
await card(['continuity-609752596743.us-central1.run.app',
            'Every number traces to the ClickHouse query that produced it.'], 5000);
await page.waitForTimeout(800);
await ctx.close();
await browser.close();
console.log(`total: ${((Date.now()-t0)/1000).toFixed(1)}s`);
