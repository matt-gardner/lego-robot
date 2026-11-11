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
function targetBox(image) {
    const box=image.target_bbox_xyxy ?? image.provenance?.target_bbox_xyxy;
    if (box == null) return null;
    if (!Array.isArray(box) || box.length!==4 || !box.every(Number.isFinite) ||
        box[0]<0 || box[1]<0 || box[2]>image.width || box[3]>image.height ||
        box[2]<=box[0] || box[3]<=box[1]) throw new Error("Invalid published target bounding box");
    return box;
  }
  function taskImage(problem,image,index,snapshot) {
    const figure=node("figure",undefined,"task-image"),frame=node("div",undefined,"target-frame");
    const img=node("img");img.loading="lazy";img.alt=`${problem.task} target in initial pile`;
    img.src=`api/buckets/image?problem=${encodeURIComponent(problem.id)}&index=${index}&snapshot=${encodeURIComponent(snapshot)}`;
    img.addEventListener("error",()=>{frame.replaceChildren(node("p","Task image unavailable / failed integrity check","missing"));});
    frame.append(img);
    let caption;
    try {
      const box=targetBox(image);
      if (box) {
        const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg"),rect=document.createElementNS(ns,"rect");
        svg.setAttribute("viewBox",`0 0 ${image.width} ${image.height}`);
        svg.setAttribute("aria-label","Visible target bounding box from renderer segmentation");
        for (const [key,value] of Object.entries({x:box[0],y:box[1],width:box[2]-box[0],height:box[3]-box[1],fill:"none",stroke:"#ffdd00","stroke-width":2,"vector-effect":"non-scaling-stroke"})) rect.setAttribute(key,value);
        svg.append(rect);frame.append(svg);
        caption="Target bounding box · visible segmentation";
      } else {
        const state=image.target_bbox_status ?? image.provenance?.target_bbox_status;
        const pixels=image.target_visible_pixels ?? image.provenance?.initial_visible_target_pixels;
        caption=state==="not_visible" || pixels===0 ? "Target not visible in this camera · no box" : "Target bounding box not measured yet";
      }
    } catch(error) {caption=error.message;}
    figure.append(frame,node("figcaption",`${image.camera || "Camera unspecified"} · initial state · ${caption}`));
    return figure;
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
    const progress=data.manifest.progress;
    if (progress) {
      status.textContent=`${problems.length} published problems. Phase: ${progress.phase || "unknown"}. Expert references: ${progress.reference_accepted ?? "unknown"} accepted / ${progress.reference_attempts ?? "unknown"} attempts (not model pass rate). Policy calibration: ${progress.policy_calibration_status || "unknown"}; ${progress.policy_rollouts ?? "unknown"} policy rollouts.`;
      if (progress.policy_rollouts===0) status.textContent+=" No policy pass rates measured yet.";
    }
    if(data.manifest.datasets) {
      status.textContent=`${problems.length} problems across ${data.manifest.datasets.length} published datasets. `+
        data.manifest.datasets.map(d=>{
          const p=d.progress;
          return p ? `${d.label || d.dataset_id || "Dataset"}: policy ${p.policy_calibration_status}; ${p.policy_rollouts} rollouts; expert references ${p.reference_accepted}/${p.reference_attempts} (not model pass rate)` :
            `${d.label || d.dataset_id || "Dataset"}: ${d.measurement_status || "unmeasured"}`;
        }).join(". ");
    }
    if(data.manifest.measurement_scopes) status.textContent+=" "+
      Object.values(data.manifest.measurement_scopes).map(s=>`${s.label}: ${s.policy_rollouts} policy rollouts (${s.status})`).join(". ");
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
        const bucketSection=node("details",undefined,"bucket-group");
        bucketSection.open=true;
        const label=(buckets.find(b=>b.id===group.bucket)?.label || group.bucket).replace(/^native:/,"").replace(/-/g," ");
        bucketSection.append(node("summary",`${label} · ${group.piles.length} piles`));
        const grid=node("div",undefined,"pile-grid");
        bucketSection.append(grid);
        container.append(bucketSection);
        for (const [pileIndex,pile] of group.piles.slice(0,remaining).entries()) {
          const pileSection=node("article",undefined,"pile-group");
          const representative=pile.tasks.find(p=>p.initial_images?.length) || pile.tasks[0];
          const scene=representative.tags?.scene || {};
          const summary=node("button",undefined,"pile-card"), preview=node("div",undefined,"pile-preview");
          summary.type="button";
          const image=representative.initial_images?.[0];
          if (image) {
          const img=node("img");img.loading="lazy";img.alt=`Initial pile ${pileIndex+1}`;
          img.src=`api/buckets/image?problem=${encodeURIComponent(representative.id)}&index=0&snapshot=${encodeURIComponent(representative.dataset_snapshot_id || data.manifest.snapshot_id)}`;
          img.addEventListener("error",()=>{img.replaceWith(node("p","Initial image unavailable / failed integrity check","missing"));});
          preview.append(img);
          } else preview.append(node("p","Initial scene images not published.","missing"));
          summary.append(preview,node("strong",`Pile ${pileIndex+1}`));
          const tags=node("div",undefined,"pile-tags");
          const descriptions=[
            scene.piece_count != null ? `${scene.piece_count} pieces` : null,
            scene.unique_piece_count != null ? `${scene.unique_piece_count} distinct shapes` : null,
            scene.unique_color_count != null ? `${scene.unique_color_count} colors` : null,
            ...Object.keys(scene.color_counts || {}),
            scene.view_obstructions?.count ? `${scene.view_obstructions.count} view obstructions` : null,
            `${pile.tasks.length} tasks`,
            ...new Set(pile.tasks.map(p=>p.task)),
          ].filter(Boolean);
          for (const description of descriptions) tags.append(node("span",description));
          summary.append(tags);
          if(image) summary.append(node("small",`Renderer: ${image.renderer || image.backend || "unspecified"}`));
          pileSection.append(summary);
          grid.append(pileSection);
          const taskContent=node("div");
          const stage=node("div",undefined,"task-stage"), controls=node("div",undefined,"carousel-controls");
          const previous=node("button","Previous"), position=node("span"), next=node("button","Next");
          const strip=node("div",undefined,"task-strip");
          strip.setAttribute("aria-label","Choose task");
          const slides=[], thumbnails=[];
          let selected=0;
          function selectTask(index) {
            selected=(index+slides.length)%slides.length;
            slides.forEach((slide,i)=>{slide.hidden=i!==selected;});
            thumbnails.forEach((thumb,i)=>thumb.setAttribute("aria-pressed",String(i===selected)));
            position.textContent=`Task ${selected+1} of ${slides.length}`;
            thumbnails[selected]?.scrollIntoView?.({block:"nearest",inline:"nearest"});
          }
          previous.addEventListener("click",()=>selectTask(selected-1));
          next.addEventListener("click",()=>selectTask(selected+1));
          strip.addEventListener("keydown",event=>{
            if(event.key==="ArrowRight" || event.key==="ArrowLeft") {
              event.preventDefault();selectTask(selected+(event.key==="ArrowRight"?1:-1));
              thumbnails[selected].focus();
            }
          });
          controls.append(previous,position,next);
          taskContent.append(stage,controls,strip);
          summary.addEventListener("click",()=>{
            const dialog=document.getElementById("pile-dialog");
            document.getElementById("pile-title").textContent=`${label} · Pile ${pileIndex+1} · ${pile.tasks.length} tasks`;
            document.getElementById("pile-tasks").replaceChildren(taskContent);
            dialog.showModal();
          });
          for (const p of pile.tasks) {
        const card=node("article",undefined,"problem");
        const target=(p.tags?.target_instance_id || p.source?.task_id || "").match(/(?:object|brick)_(\d+)/);
        const dimensions=p.tags?.target?.part_dimensions_m;
        const targetLabel=target ? `Object ${Number(target[1])+1}` : "Target";
        const taskTitle=`${p.task} · ${targetLabel}${dimensions ? " · "+dimensions.map(x=>(x*1000).toFixed(1)).join(" × ")+" mm" : ""}`;
        const thumb=node("button",undefined,"task-thumbnail");
        thumb.type="button";thumb.setAttribute("aria-label",taskTitle);
        const thumbnailImage=p.initial_images?.[0];
        if(thumbnailImage) thumb.append(taskImage(p,thumbnailImage,0,p.dataset_snapshot_id || data.manifest.snapshot_id));
        thumb.append(node("span",taskTitle));
        const slideIndex=slides.length;
        thumb.addEventListener("click",()=>selectTask(slideIndex));
        thumbnails.push(thumb);strip.append(thumb);
        for (const [index,image] of (p.initial_images || []).entries()) {
          card.append(taskImage(p,image,index,p.dataset_snapshot_id || data.manifest.snapshot_id));
        }
        if (!p.initial_images?.length) card.append(node("p","Task image not published yet","missing"));
        card.append(node("h3",taskTitle));
        card.append(node("p",`Feasibility: ${p.feasibility?.status || "unmeasured"}. Low policy success does not prove infeasibility or solvability.`));
        const measured=groupFor(p);
        stats(card,measured);
        const failures=measured.reduce((n,g)=>n+Math.max(0,(g.attempts ?? g.trajectories?.length ?? 0)-(g.successes ?? 0)),0);
        card.append(node("p",`Observed unsuccessful rollouts: ${measured.length ? failures : "unmeasured"}. Feasibility vs policy failure attribution: ${p.failure_attribution_status || "unmeasured"}. Visibility / reach: ${p.tags?.visibility ?? "unmeasured"} / ${p.tags?.reach ?? "unmeasured"}.`));
        json(card,"Tags / provenance / measurement and failure evidence",p);
        json(card,"Raw measured groups (invalid actions, policy vs feasibility failures)",groupFor(p));
        slides.push(card);stage.append(card);
          }
          selectTask(0);
        }
        remaining=Math.max(0,remaining-group.piles.length);
      }
      document.getElementById("more").hidden=shown>=pileCount;
    }
    for (const id of ["bucket","task","search"]) document.getElementById(id).addEventListener("input",()=>{shown=24;render();});
    document.getElementById("more").addEventListener("click",()=>{shown+=24;render();});
    render();
    if(typeof setInterval!=="undefined") setInterval(async()=>{
      if(document.getElementById("pile-dialog").open) return;
      try {
        const latest=await fetch("api/buckets");
        const publication=await latest.json();
        if(!latest.ok) throw new Error(publication.error || `HTTP ${latest.status}`);
        if(publication.manifest.snapshot_id!==data.manifest.snapshot_id) location.reload();
      } catch(error) {status.textContent=`Publication refresh failed: ${error.message}`;}
    },20000);
  } catch(error) {status.textContent=`Bucket publication unavailable: ${error.message}`;}
}
if (typeof document !== "undefined") {
  document.getElementById("pile-close").addEventListener("click",()=>document.getElementById("pile-dialog").close());
  document.getElementById("pile-dialog").addEventListener("click",event=>{
    if(event.target===document.getElementById("pile-dialog")) {
      const bounds=event.target.getBoundingClientRect();
      if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom) event.target.close();
    }
  });
  document.getElementById("refresh").addEventListener("click",()=>location.reload());
  start();
}
if (typeof module !== "undefined") module.exports={wilson,aggregate,groupPiles,targetBox};
