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
function bucketLabel(value) {
  return value.replace(/^current:/,"").replace(/-/g," ");
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
      const annotations=image.target_annotations ?? image.provenance?.target_annotations ??
        [{...image,role:"source_object"}];
      if (!Array.isArray(annotations) || !annotations.length) throw new Error("Invalid published target annotations");
      const captions=[];
      for (const annotation of annotations) {
      if (!["source_object","destination"].includes(annotation.role)) throw new Error("Invalid published target annotation role");
      const label=annotation.role==="destination" ? "Destination goal marker (cyan)" : "Source object (yellow)";
      const box=targetBox({...annotation,width:image.width,height:image.height});
      if (box) {
        const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg"),rect=document.createElementNS(ns,"rect");
        svg.setAttribute("viewBox",`0 0 ${image.width} ${image.height}`);
        svg.setAttribute("aria-label","Visible target bounding box from renderer segmentation");
        const title=document.createElementNS(ns,"title");title.textContent=label;rect.append(title);
        for (const [key,value] of Object.entries({x:box[0],y:box[1],width:box[2]-box[0],height:box[3]-box[1],fill:"none",stroke:annotation.role==="destination" ? "#00e5ff" : "#ffdd00","stroke-width":2,"vector-effect":"non-scaling-stroke"})) rect.setAttribute(key,value);
        svg.append(rect);frame.append(svg);
        captions.push(`${label} · visible segmentation`);
      } else {
        const state=annotation.target_bbox_status ?? annotation.provenance?.target_bbox_status;
        const pixels=annotation.target_visible_pixels ?? annotation.provenance?.initial_visible_target_pixels;
        captions.push(`${label} · ${state==="not_visible" || pixels===0 ? "not visible in this camera · no box" : "bounding box not measured yet"}`);
      }
      }
      if (["push","place"].includes(problem.task) && !annotations.some(a=>a.role==="destination")) captions.push("Destination bounding box not measured yet");
      caption=captions.join(" · ");
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
function json(parent, label, value, remember) {
  const d = node("details"); d.append(node("summary", label), node("pre", JSON.stringify(value ?? "Unmeasured / missing", null, 2))); parent.append(d);
  remember?.(d);
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
function bucketChart(parent, series) {
  if(!series.some(s=>s.cells.length)) {
    parent.append(node("p","Unmeasured: no policy success rates yet.","missing"));return;
  }
  const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");
  svg.setAttribute("viewBox","0 0 360 220");svg.setAttribute("class","bucket-chart");
  svg.setAttribute("role","img");svg.setAttribute("aria-label","Success rate versus assistance percent, by task");
  function shape(tag,attributes,text) {
    const e=document.createElementNS(ns,tag);
    for(const [key,value] of Object.entries(attributes)) e.setAttribute(key,value);
    if(text!==undefined)e.textContent=text;
    svg.append(e);return e;
  }
  for(const percent of [0,50,100]) {
    const x=45+percent*2.85,y=175-percent*1.4;
    shape("line",{x1:45,x2:330,y1:y,y2:y,stroke:"#8885"});
    shape("text",{x:38,y:y+4,"text-anchor":"end",fill:"currentColor","font-size":11},`${percent}%`);
    shape("text",{x,y:192,"text-anchor":"middle",fill:"currentColor","font-size":11},`${percent}%`);
  }
  shape("text",{x:185,y:215,"text-anchor":"middle",fill:"currentColor","font-size":12},"Assistance");
  shape("text",{x:45,y:18,fill:"currentColor","font-size":12},"Success rate");
  const colors=["#54b5ff","#ffbd59","#71db99","#e296ef"];
  const marks=[];
  let hovered=null,focused=null;
  function highlight() {
    const active=hovered ?? focused;
    for(const mark of marks) {
      const selected=active===mark.task,muted=active!==null&&!selected;
      mark.point.setAttribute("fill",muted ? "#888" : mark.color);
      mark.point.setAttribute("opacity",muted ? .25 : 1);
      mark.point.setAttribute("r",selected ? 6 : 4);
      mark.bar.setAttribute("stroke",muted ? "#888" : mark.color);
      mark.bar.setAttribute("opacity",muted ? .25 : 1);
    }
  }
  series.forEach((s,index)=>{
    const color=colors[index%colors.length];
    for(const c of s.cells) {
      const ci=wilson(c.successes,c.n),x=45+c.assistance*285,y=175-c.successes/c.n*140;
      const bar=shape("line",{x1:x,x2:x,y1:175-ci[0]*140,y2:175-ci[1]*140,stroke:color,"stroke-width":1});
      const point=shape("circle",{cx:x,cy:y,r:4,fill:color});
      marks.push({task:s.task,color,point,bar});
      const title=document.createElementNS(ns,"title");
      title.textContent=`${s.task}: ${(c.assistance*100).toFixed(2)}% assistance, ${c.successes}/${c.n} successes (${(100*c.successes/c.n).toFixed(1)}%), ${c.groups} groups`;
      point.append(title);
    }
    const legend=node("span",`${s.task}${s.cells.length ? "" : " (unmeasured)"}`,"chart-legend");
    legend.style=`color:${color}`;legend.setAttribute("tabindex","0");
    legend.addEventListener("mouseenter",()=>{hovered=s.task;highlight();});
    legend.addEventListener("mouseleave",()=>{hovered=null;highlight();});
    legend.addEventListener("focus",()=>{focused=s.task;highlight();});
    legend.addEventListener("blur",()=>{focused=null;highlight();});
    parent.append(legend);
  });
  parent.append(svg,node("small","Points: sampled problems only; bars: descriptive Wilson 95% intervals. No interpolation."));
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
  const preferenceKey="lego-buckets-view-v1";
  let preferences={};
  try {
    const saved=JSON.parse(localStorage.getItem(preferenceKey) || "{}");
    if(saved && typeof saved==="object" && !Array.isArray(saved)) preferences=saved;
  } catch(error) {console.warn("Could not restore bucket dashboard preferences",error);}
  function save() {
    try {localStorage.setItem(preferenceKey,JSON.stringify(preferences));}
    catch(error) {console.warn("Could not save bucket dashboard preferences",error);}
  }
  function rememberDetails(element,key,defaultOpen=false) {
    element.open=typeof preferences[key]==="boolean" ? preferences[key] : defaultOpen;
    element.addEventListener("toggle",()=>{preferences[key]=element.open;save();});
  }
  try {
    const response=await fetch("api/buckets");
    const data=await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    const problems=data.problems.problems.filter(p=>p.source?.kind!=="legacy");
    const groups=data.calibration.groups;
    const admitted=data.admitted_piles || [];
    const admissionBucket=p=>`current:${p.tags.piece_count}-pieces`;
    function admissionCard(pile) {
        const card=node("article",undefined,"pile-group");
        card.append(node("h3",`${pile.tags.piece_count} pieces · admitted pile`));
        if(pile.initial_images.length) {
          const image=node("img");image.loading="lazy";image.alt="Admitted pile preview";
          image.style="width:100%;height:auto";
          image.src=`api/buckets/image?problem=${encodeURIComponent(pile.id)}&index=0&snapshot=${encodeURIComponent(pile.dataset_snapshot_id)}`;
          image.addEventListener("error",()=>image.replaceWith(node("p","Pile preview unavailable / failed integrity check","missing")));
          card.append(image);
        } else card.append(node("p","Pile preview not rendered yet","missing"));
        card.append(node("p","Admitted pile · task pool pending · policy rates unmeasured."));
        return card;
    }
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
          const protocol=d.protocol?.horizon ? ` (${d.protocol.horizon}-step ${d.protocol.measurement_scope || "screening"})` : "";
          return p ? `${d.label || d.dataset_id || "Dataset"}${protocol}: policy ${p.policy_calibration_status}; ${p.policy_rollouts} rollouts; expert references ${p.reference_accepted ?? "unmeasured"}/${p.reference_attempts ?? "unmeasured"} (not model pass rate)` :
            `${d.label || d.dataset_id || "Dataset"}: ${d.measurement_status || "unmeasured"}`;
        }).join(". ");
    }
    if(data.manifest.measurement_scopes) status.textContent+=" "+
      Object.values(data.manifest.measurement_scopes).map(s=>`${s.label}: ${s.policy_rollouts} policy rollouts (${s.status})`).join(". ");
    const sourcePath=s=>typeof s==="object" ? s?.path : s;
    const measuredGroups=new Map();
    const groupFor=p=>{
      if(!measuredGroups.has(p.id)) measuredGroups.set(p.id,groups.filter(g=>g.problem_id ? g.problem_id===p.id : (g.bucket_id ?? g.bucket)===p.bucket && g.task===p.task && sourcePath(g.source)===sourcePath(p.source) && g.problem_index===p.problem_index));
      return measuredGroups.get(p.id);
    };
    const bucketIds=[...new Set([...buckets.map(b=>b.id),...problems.map(p=>p.bucket),...admitted.map(admissionBucket)])].sort();
    const overview=node("table"), heading=node("tr");
    for (const label of ["Bucket", "Problems", "Success vs assistance", "Measured coverage by task", "Initial images"]) heading.append(node("th",label));
    overview.append(heading);
    for (const id of bucketIds) {
      const b=buckets.find(x=>x.id===id), ps=problems.filter(p=>p.bucket===id);
      const tasks=[...new Set([...(b?.tasks || []).map(t=>t.task),...ps.map(p=>p.task)])].sort();
      const row=node("tr"),chart=node("td"),coverage=node("td");
      function draw() {
      chart.replaceChildren();coverage.replaceChildren();
      bucketChart(chart,tasks.map(task=>({task,cells:aggregate(ps.filter(p=>p.task===task).flatMap(groupFor))})));
      for(const task of tasks) {
        const tp=ps.filter(p=>p.task===task), measured=tp.filter(p=>aggregate(groupFor(p)).length);
        const rollouts=aggregate(tp.flatMap(groupFor)).reduce((n,c)=>n+c.n,0);
        coverage.append(node("p",`${task}: ${measured.length}/${tp.length} problems · ${rollouts} rollouts`));
      }
      const ap=admitted.filter(p=>admissionBucket(p)===id);
      if(ap.length) coverage.append(node("p",`${ap.length} admitted piles · task pool pending`));
      }
      draw();
      const ap=admitted.filter(p=>admissionBucket(p)===id);
      row.append(node("td",bucketLabel(b?.label || id)),node("td",ps.length ? String(ps.length) : "Tasks pending"),chart,coverage,
        node("td",ap.length ? `${ps.filter(p=>p.initial_images?.length).length}/${ps.length} task images; ${ap.filter(p=>p.initial_images.length).length}/${ap.length} admitted pile previews` : `${ps.filter(p=>p.initial_images?.length).length}/${ps.length}`));
          row.addEventListener("click",()=>{
            document.getElementById("bucket").value=id;
            document.getElementById("task").value="";
            shown=24;saveFilters();render();
          });
          overview.append(row);
      const opt=node("option",bucketLabel(b?.label || id));opt.value=id;document.getElementById("bucket").append(opt);
    }
    document.getElementById("populations").append(overview);
    for(const pilot of data.distance_pilots || []) {
      const section=node("details");
      section.append(node("summary",`Distance-assisted pilot · ${pilot.child_count} child starts / ${pilot.parent_count} parent tasks`));
      section.append(node("p","Physically assisted starts, not original unassisted trials. Expert-prefix percentage does not measure remaining task distance."));
      const table=node("table"),header=node("tr");
      for(const label of ["Task","Active stage","Remaining stages","Within-stage distance","Policy success","Horizon"])header.append(node("th",label));
      table.append(header);
      for(const problem of pilot.problems) {
        const distance=problem.tags.task_assistance.distance;
        const measured=pilot.groups.filter(g=>g.problem_id===problem.id);
        const cells=aggregate(measured);
        const attempts=cells.reduce((n,c)=>n+c.n,0),successes=cells.reduce((n,c)=>n+c.successes,0);
        const metric=distance.within_stage;
        const horizons=[...new Set(measured.map(g=>g.protocol?.horizon || pilot.manifest.protocol?.horizon).filter(Boolean))];
        const row=node("tr");
        for(const value of [problem.task,distance.active_stage,distance.remaining_stages.join(" → "),
          `${metric.metric}: ${metric.value} ${metric.units}`,
          attempts ? `${successes}/${attempts} (${(100*successes/attempts).toFixed(1)}%)` : "Unmeasured",
          horizons.length ? horizons.join(", ") : "Unmeasured"])row.append(node("td",value));
        table.append(row);
      }
      section.append(table);
      json(section,"Parent, reset and measurement provenance",pilot);
      document.getElementById("distance-pilots").append(section);
    }
    for (const task of [...new Set(problems.map(p=>p.task))].sort()) {const o=node("option",task);o.value=task;document.getElementById("task").append(o);}
    for (const id of ["bucket","task","search"]) {
      const element=document.getElementById(id);
      if(typeof preferences[id]==="string" &&
        (id==="search" || (id==="bucket" ? bucketIds : [...new Set(problems.map(p=>p.task))]).includes(preferences[id]))) element.value=preferences[id];
    }
    let shown=Number.isInteger(preferences.shown) && preferences.shown>=24 ? preferences.shown : 24;
    function saveFilters() {
      for (const id of ["bucket","task","search"]) preferences[id]=document.getElementById(id).value;
      preferences.shown=shown;save();
    }
    function render() {
      const bucket=document.getElementById("bucket").value,task=document.getElementById("task").value,q=document.getElementById("search").value.toLowerCase();
      const filtered=problems.filter(p=>(!bucket||p.bucket===bucket)&&(!task||p.task===task)&&(!q||JSON.stringify(p).toLowerCase().includes(q)));
      const container=document.getElementById("problems");container.replaceChildren();
      const hierarchy=groupPiles(filtered);
      const filteredAdmissions=admitted.filter(p=>(!bucket||admissionBucket(p)===bucket)&&!task&&(!q||JSON.stringify(p).toLowerCase().includes(q)));
      for(const pile of filteredAdmissions) {
        const id=admissionBucket(pile);
        let group=hierarchy.find(b=>b.bucket===id);
        if(!group){group={bucket:id,piles:[]};hierarchy.push(group);}
        group.piles.push({id:pile.id,admission:pile});
      }
      hierarchy.sort((a,b)=>a.bucket.localeCompare(b.bucket));
      const pileCount=hierarchy.reduce((n,b)=>n+b.piles.length,0);
      document.getElementById("count").textContent=`${filtered.length} matching problems in ${pileCount} piles; showing ${Math.min(shown,pileCount)} piles.`;
      let remaining=shown;
      for (const group of hierarchy) {
        if (!remaining) break;
        const bucketSection=node("details",undefined,"bucket-group");
        rememberDetails(bucketSection,`bucket-open:${group.bucket}`,true);
        const label=bucketLabel(buckets.find(b=>b.id===group.bucket)?.label || group.bucket).replace(/^native:/,"");
        bucketSection.append(node("summary",`${label} · ${group.piles.length} piles`));
        const grid=node("div",undefined,"pile-grid");
        bucketSection.append(grid);
        container.append(bucketSection);
        for (const [pileIndex,pile] of group.piles.slice(0,remaining).entries()) {
          if(pile.admission){grid.append(admissionCard(pile.admission));continue;}
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
            preferences[`selected-task:${group.bucket}:${pile.id}`]=pile.tasks[selected].id;save();
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
            buildTasks();
            const dialog=document.getElementById("pile-dialog");
            document.getElementById("pile-title").textContent=`${label} · Pile ${pileIndex+1} · ${pile.tasks.length} tasks`;
            document.getElementById("pile-tasks").replaceChildren(taskContent);
            dialog.showModal();
          });
          let built=false;
          function buildTasks() {
          if(built)return;
          built=true;
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
        json(card,"Tags / provenance / measurement and failure evidence",p,d=>rememberDetails(d,`task-provenance:${p.id}`));
        json(card,"Raw measured groups (invalid actions, policy vs feasibility failures)",groupFor(p),d=>rememberDetails(d,`task-groups:${p.id}`));
        slides.push(card);stage.append(card);
          }
          const restored=pile.tasks.findIndex(p=>p.id===preferences[`selected-task:${group.bucket}:${pile.id}`]);
          selectTask(restored<0 ? 0 : restored);
          }
        }
        remaining=Math.max(0,remaining-group.piles.length);
      }
      document.getElementById("more").hidden=shown>=pileCount;
    }
    for (const id of ["bucket","task","search"]) document.getElementById(id).addEventListener("input",()=>{shown=24;saveFilters();render();});
    document.getElementById("more").addEventListener("click",()=>{shown+=24;saveFilters();render();});
    render();
    if(typeof setInterval!=="undefined") setInterval(async()=>{
      if(document.getElementById("pile-dialog").open) return;
      try {
        const latest=await fetch("api/buckets");
        const publication=await latest.json();
        if(!latest.ok) throw new Error(publication.error || `HTTP ${latest.status}`);
        if(publication.manifest.snapshot_id!==data.manifest.snapshot_id) location.reload();
      } catch(error) {status.hidden=false;status.textContent=`Publication refresh failed: ${error.message}`;}
    },20000);
  } catch(error) {status.hidden=false;status.textContent=`Bucket publication unavailable: ${error.message}`;}
}
if (typeof document !== "undefined") {
  document.getElementById("pile-close").addEventListener("click",()=>document.getElementById("pile-dialog").close());
  document.getElementById("pile-dialog").addEventListener("click",event=>{
    if(event.target===document.getElementById("pile-dialog")) {
      const bounds=event.target.getBoundingClientRect();
      if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom) event.target.close();
    }
  });
  start();
}
if (typeof module !== "undefined") module.exports={wilson,aggregate,groupPiles,targetBox};
