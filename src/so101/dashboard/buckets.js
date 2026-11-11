/* global document, fetch */
"use strict";
function wilson(successes, n) {
  if (!n) return null;
  const z = 1.96, p = successes / n, d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
function aggregate(groups) {
  const cells = new Map();
  for (const g of groups) {
    const n = g.attempts ?? g.trajectories?.length;
    if (!Number.isInteger(n) || n <= 0 || !Number.isInteger(g.successes) || g.successes < 0 || g.successes > n) continue;
    const key = g.assistance;
    if (typeof key !== "number" || !Number.isFinite(key)) continue;
    const c = cells.get(key) || {assistance: key, n: 0, successes: 0, groups: 0, mixed: 0};
    c.n += n; c.successes += g.successes; c.groups++;
    if (g.successes > 0 && g.successes < n) c.mixed++;
    cells.set(key, c);
  }
  return [...cells.values()].sort((a, b) => a.assistance - b.assistance);
}
function pileIdentity(problem) {
  return problem.source?.scene_id || problem.pile_id || problem.scene_id || problem.id;
}
function groupPiles(problems) {
  const buckets = new Map();
  for (const problem of problems) {
    if (!buckets.has(problem.bucket)) buckets.set(problem.bucket,new Map());
    const piles=buckets.get(problem.bucket), id=pileIdentity(problem);
    if (!piles.has(id)) piles.set(id,[]);
    piles.get(id).push(problem);
  }
  return [...buckets].sort(([a],[b])=>a.localeCompare(b)).map(([bucket,piles])=>({
    bucket, piles:[...piles].map(([id,tasks])=>({id,tasks})),
  }));
}
function node(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function json(parent, label, value) {
  const d = node("details"); d.append(node("summary", label), node("pre", JSON.stringify(value ?? "Unmeasured / missing", null, 2))); parent.append(d);
}
function curve(parent, cells) {
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 500 150"); svg.setAttribute("class", "curve");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Policy success versus assistance, with app-derived Wilson 95 percent intervals");
  for (const c of cells) {
    const ci = wilson(c.successes, c.n), x = 25 + c.assistance * 450;
    const line = document.createElementNS(ns, "line");
    for (const [k,v] of Object.entries({x1:x,x2:x,y1:130-ci[0]*110,y2:130-ci[1]*110,stroke:"currentColor"})) line.setAttribute(k,v);
    const point = document.createElementNS(ns,"circle");
    for (const [k,v] of Object.entries({cx:x,cy:130-c.successes/c.n*110,r:4,fill:"currentColor"})) point.setAttribute(k,v);
    svg.append(line,point);
  }
  parent.append(svg);
}
function stats(parent, groups) {
  const cells = aggregate(groups);
  if (!cells.length) { parent.append(node("p", "Unmeasured: no valid rollout counts published.", "missing")); return; }
  curve(parent,cells);
  const table = node("table"), head = node("tr");
  for (const s of ["Assistance", "Policy success", "95% CI", "Rollouts / groups", "Mixed groups"]) head.append(node("th",s));
  table.append(head);
  for (const c of cells) {
    const ci = wilson(c.successes,c.n), row=node("tr");
    for (const s of [c.assistance, `${c.successes}/${c.n}`, ci.map(x=>(100*x).toFixed(1)+"%").join("–"), `${c.n} / ${c.groups}`, `${c.mixed}/${c.groups}`]) row.append(node("td",String(s)));
    table.append(row);
  }
  parent.append(table, node("p","Intervals: app-derived Wilson 95%, descriptive rollout counts; repeated seeds/group correlation are not modeled. Mixed means both successes and failures within the same group, not aggregate 50% success."));
}
async function start() {
  const status=document.getElementById("status");
  try {
    const response=await fetch("api/buckets");
    const data=await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    const problems=data.problems.problems.filter(p=>p.source?.kind!=="legacy");
    const groups=data.calibration.groups;
    const buckets=(data.problems.buckets || []).filter(b=>!b.id.startsWith("corpus:"));
    const pilot=data.calibration.qualification_status !== "qualified";
    status.textContent=`${pilot ? "SCREENING ONLY — not calibrated qualification" : "Measurement status: "+(data.calibration.qualification_status || "Unqualified / unspecified")}. ${problems.length} published problems. ${data.calibration.complete ? "Measurement plan complete" : "Measurements in progress"}.`;
    document.getElementById("provenance").textContent=JSON.stringify({manifest:data.manifest,protocol:data.calibration.protocol},null,2);
    const sourcePath=s=>typeof s==="object" ? s?.path : s;
    const groupFor=p=>groups.filter(g=>g.problem_id ? g.problem_id===p.id : (g.bucket_id ?? g.bucket)===p.bucket && g.task===p.task && sourcePath(g.source)===sourcePath(p.source) && g.problem_index===p.problem_index);
    const bucketIds=[...new Set([...buckets.map(b=>b.id),...problems.map(p=>p.bucket)])].sort();
    const overview=node("table"), heading=node("tr");
    for (const label of ["Bucket", "Task", "Problems", "Assistance tested", "Success", "95% CI", "Rollouts / groups", "Measured problems", "Mixed groups", "Per-problem success range", "Initial images"]) heading.append(node("th",label));
    overview.append(heading);
    for (const id of bucketIds) {
      const b=buckets.find(x=>x.id===id), ps=problems.filter(p=>p.bucket===id);
      const tasks=[...new Set([...(b?.tasks || []).map(t=>t.task),...ps.map(p=>p.task)])].sort();
      for (const task of tasks) {
        const tp=ps.filter(p=>p.task===task), declared=b?.tasks?.find(t=>t.task===task)?.population_total;
        const cells=aggregate(tp.flatMap(groupFor));
        for (const c of cells.length ? cells : [null]) {
          const rates=c ? tp.flatMap(p=>aggregate(groupFor(p)).filter(x=>x.assistance===c.assistance).map(x=>x.successes/x.n)) : [];
          const row=node("tr"), ci=c ? wilson(c.successes,c.n) : null;
          const values=[b?.label || id,task,declared ?? `${tp.length} enumerated`,
            c ? c.assistance : "Unmeasured",
            c ? `${(100*c.successes/c.n).toFixed(1)}% (${c.successes}/${c.n})` : "—",
            ci ? ci.map(x=>(100*x).toFixed(1)+"%").join("–") : "—",
            c ? `${c.n} / ${c.groups}` : "0 / 0",
            `${rates.length}/${tp.length}`,
            c ? `${c.mixed}/${c.groups}` : "—",
            rates.length ? `${(Math.min(...rates)*100).toFixed(0)}–${(Math.max(...rates)*100).toFixed(0)}%` : "—",
            `${tp.filter(p=>p.initial_images?.length).length}/${tp.length}`];
          for (const value of values) row.append(node("td",String(value)));
          row.addEventListener("click",()=>{
            document.getElementById("bucket").value=id;
            document.getElementById("task").value=task;
            shown=24;render();
          });
          overview.append(row);
        }
      }
      const opt=node("option",id);opt.value=id;document.getElementById("bucket").append(opt);
    }
    document.getElementById("populations").append(overview);
    for (const task of [...new Set(problems.map(p=>p.task))].sort()) {const o=node("option",task);o.value=task;document.getElementById("task").append(o);}
    let shown=24;
    function render() {
      const bucket=document.getElementById("bucket").value,task=document.getElementById("task").value,q=document.getElementById("search").value.toLowerCase();
      const filtered=problems.filter(p=>(!bucket||p.bucket===bucket)&&(!task||p.task===task)&&(!q||JSON.stringify(p).toLowerCase().includes(q)));
      const container=document.getElementById("problems");container.replaceChildren();
      const hierarchy=groupPiles(filtered), pileCount=hierarchy.reduce((n,b)=>n+b.piles.length,0);
      document.getElementById("count").textContent=`${filtered.length} matching problems in ${pileCount} piles; showing ${Math.min(shown,pileCount)} piles.`;
      let remaining=shown;
      for (const group of hierarchy) {
        if (!remaining) break;
        const bucketSection=node("section",undefined,"bucket-group");
        bucketSection.append(node("h2",buckets.find(b=>b.id===group.bucket)?.label || group.bucket));
        container.append(bucketSection);
        for (const pile of group.piles.slice(0,remaining)) {
          const pileSection=node("details",undefined,"pile-group");
          pileSection.open=true;
          pileSection.append(node("summary",`Pile ${pile.id} · ${pile.tasks.length} tasks`));
          bucketSection.append(pileSection);
          for (const p of pile.tasks) {
        const card=node("details",undefined,"problem");
        card.append(node("summary",`${p.task} · ${p.source?.task_id || p.id}`));
        const images=node("div",undefined,"images");
        for (const [i,image] of (p.initial_images || []).entries()) {
          const f=node("figure"),img=node("img");img.loading="lazy";img.alt=`Initial ${image.camera || "scene"} for ${p.id}`;
          img.src=`api/buckets/image?problem=${encodeURIComponent(p.id)}&index=${i}&snapshot=${encodeURIComponent(data.manifest.snapshot_id)}`;
          img.addEventListener("error",()=>{img.replaceWith(node("p","Initial image unavailable / failed integrity check","missing"));});
          f.append(img,node("figcaption",image.camera || "Initial scene"));images.append(f);
        }
        card.append(images);
        if (!p.initial_images?.length) card.append(node("p","Initial scene images not published.", "missing"));
        card.append(node("p",`Feasibility: ${p.feasibility?.status || "unmeasured"}. Low policy success does not prove infeasibility or solvability.`));
        const measured=groupFor(p);
        stats(card,measured);
        const failures=measured.reduce((n,g)=>n+Math.max(0,(g.attempts ?? g.trajectories?.length ?? 0)-(g.successes ?? 0)),0);
        card.append(node("p",`Observed unsuccessful rollouts: ${measured.length ? failures : "unmeasured"}. Feasibility vs policy failure attribution: ${p.failure_attribution_status || "unmeasured"}. Visibility / reach: ${p.tags?.visibility ?? "unmeasured"} / ${p.tags?.reach ?? "unmeasured"}.`));
        json(card,"Tags / provenance / measurement and failure evidence",p);
        json(card,"Raw measured groups (invalid actions, policy vs feasibility failures)",groupFor(p));
        pileSection.append(card);
          }
        }
        remaining=Math.max(0,remaining-group.piles.length);
      }
      document.getElementById("more").hidden=shown>=pileCount;
    }
    for (const id of ["bucket","task","search"]) document.getElementById(id).addEventListener("input",()=>{shown=24;render();});
    document.getElementById("more").addEventListener("click",()=>{shown+=24;render();});
    render();
  } catch(error) {status.textContent=`Bucket publication unavailable: ${error.message}`;}
}
if (typeof document !== "undefined") {
  document.getElementById("refresh").addEventListener("click",()=>location.reload());
  start();
}
if (typeof module !== "undefined") module.exports={wilson,aggregate,groupPiles};
