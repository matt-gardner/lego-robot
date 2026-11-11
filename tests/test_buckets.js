const {test} = require("node:test");
const assert = require("node:assert/strict");
const {wilson,aggregate,groupPiles,targetBox} = require("../src/so101/dashboard/buckets.js");
const vm = require("node:vm");
const fs = require("node:fs");
test("four successes do not imply certain success or calibration",()=>{
  const ci=wilson(4,4);
  assert.ok(ci[0]<0.52 && ci[0]>0.50);
  assert.ok(ci[1]<=1);
  assert.equal(wilson(0,0),null);
});
test("target boxes are image-bound measurements, never guessed",()=>{
  assert.deepEqual(targetBox({width:320,height:240,target_bbox_xyxy:[12,20,100,150]}),[12,20,100,150]);
  assert.equal(targetBox({width:320,height:240,target_bbox_status:"not_visible"}),null);
  assert.throws(()=>targetBox({width:320,height:240,target_bbox_xyxy:[0,0,500,50]}),/Invalid/);
});
test("mixed group yield differs from pooled half-success",()=>{
  const cells=aggregate([
    {assistance:0,successes:4,attempts:4},
    {assistance:0,successes:0,attempts:4},
    {assistance:0.5,successes:2,trajectories:[{},{},{},{}]},
    {assistance:0,successes:8,attempts:4},
  ]);
  assert.equal(cells[0].n,8);
  assert.equal(cells[0].mixed,0);
  assert.equal(cells[1].mixed,1);
});
test("hierarchy groups target tasks by exact scene within each bucket",()=>{
  const grouped=groupPiles([
    {id:"a",bucket:"native3",source:{scene_id:"pile1"}},
    {id:"b",bucket:"native3",source:{scene_id:"pile1"}},
    {id:"c",bucket:"native3",source:{scene_id:"pile2"}},
    {id:"d",bucket:"native5",source:{scene_id:"pile1"}},
  ]);
  assert.equal(grouped.length,2);
  assert.equal(grouped[0].piles.length,2);
  assert.deepEqual(grouped[0].piles[0].tasks.map(p=>p.id),["a","b"]);
  assert.equal(grouped[1].piles[0].tasks[0].id,"d");
});
test("page enumerates unmeasured tasks and binds initial images to snapshot",async()=>{
  class Element {
    constructor(){this.children=[];this.value="";this.listeners={};this.textContent="";}
    append(...children){this.children.push(...children);}
    replaceChildren(...children){this.children=children;}
    setAttribute(key,value){this[key]=value;}
    addEventListener(key,fn){this.listeners[key]=fn;}
    showModal(){this.open=true;}
    close(){this.open=false;}
  }
  const elements=new Map();
  const saved=new Map();
  const localStorage={getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)};
  const document={
    getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
    createElement(){return new Element();},
    createElementNS(){return new Element();},
  };
  const payload={
    manifest:{snapshot_id:"snapshot"},
    problems:{buckets:[{id:"b",population_total:2,tasks:[{task:"push",population_total:1},{task:"place",population_total:1}]},
        {id:"corpus:old",population_total:1,tasks:[{task:"grasp",population_total:1}]}],
      problems:[{id:"first",bucket:"b",task:"push",initial_images:[{camera:"wrist",width:320,height:240,
        target_annotations:[
          {role:"source_object",target_bbox_xyxy:[12,20,100,150]},
          {role:"destination",entity_kind:"goal_marker",target_bbox_xyxy:[200,100,230,130]},
        ]}]},
        {id:"second",bucket:"b",task:"place",initial_images:[]},
        {id:"old",bucket:"corpus:old",task:"grasp",source:{kind:"legacy"},initial_images:[]}]},
    calibration:{protocol:{rollouts_per_group:4},groups:[{problem_id:"first",assistance:0,attempts:4,successes:4}]},
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,localStorage,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(elements.get("status").textContent,/SCREENING ONLY/);
  assert.match(elements.get("count").textContent,/2 matching problems/);
  assert.equal(elements.get("problems").children.length,1);
  const bucketSection=elements.get("problems").children[0];
  assert.equal(bucketSection.open,true);
  const pileCard=bucketSection.children[1].children[0];
  assert.equal(pileCard.className,"pile-group");
  assert.ok(!pileCard.open);
  assert.equal(pileCard.children.length,1);
  pileCard.children[0].listeners.click();
  assert.equal(elements.get("pile-dialog").open,true);
  const carousel=elements.get("pile-tasks").children[0];
  assert.equal(carousel.children[0].className,"task-stage");
  assert.equal(carousel.children[0].children[0].hidden,false);
  assert.equal(carousel.children[2].className,"task-strip");
  assert.equal(carousel.children[2].children[0]["aria-pressed"],"true");
  const all=[];
  function collect(e){all.push(e);for(const child of e.children || [])collect(child);}
  collect(elements.get("problems"));
  collect(elements.get("pile-tasks"));
  bucketSection.children[1].children[1].children[0].listeners.click();
  collect(elements.get("pile-tasks"));
  assert.ok(all.some(e=>e.textContent==="Pile 1"));
  assert.ok(all.some(e=>e.textContent==="push · Target"));
  assert.ok(all.some(e=>e["aria-label"]==="Visible target bounding box from renderer segmentation"));
  assert.ok(all.some(e=>e.x===12 && e.y===20 && e.width===88 && e.height===130));
  assert.ok(all.some(e=>e.x===200 && e.y===100 && e.width===30 && e.height===30 && e.stroke==="#00e5ff"));
  assert.ok(all.some(e=>e.textContent.includes("Destination goal marker (cyan) · visible segmentation")));
  assert.ok(all.some(e=>e.src==="api/buckets/image?problem=first&index=0&snapshot=snapshot"));
  assert.ok(all.some(e=>e.textContent==="Initial scene images not published."));
  assert.ok(all.some(e=>e.textContent==="Unmeasured: no valid rollout counts published."));
  const overview=elements.get("populations").children[0];
  assert.equal(overview.children.length,2);
  assert.equal(elements.get("bucket").children.length,1);
  assert.equal(elements.get("task").children.length,2);
  assert.ok(!all.some(e=>e.textContent.includes("corpus:old")));
  const measured=overview.children[1];
  assert.equal(measured.children[1].textContent,"2");
  assert.ok(measured.children[3].children.some(e=>e.textContent==="push: 1/1 problems · 4 rollouts"));
  assert.ok(measured.children[3].children.some(e=>e.textContent==="place: 0/1 problems · 0 rollouts"));
  const plotted=[];
  function plotNodes(e){plotted.push(e);for(const child of e.children || [])plotNodes(child);}
  plotNodes(measured.children[2]);
  assert.ok(plotted.some(e=>e.cx===45 && e.cy===35));
  assert.ok(plotted.some(e=>e.textContent==="Assistance"));
  assert.ok(plotted.some(e=>e.textContent==="Success rate"));
  measured.listeners.click();
  assert.equal(elements.get("bucket").value,"b");
  assert.match(elements.get("count").textContent,/2 matching problems/);
  payload.manifest.progress={phase:"expert_reference_qualification",reference_attempts:35,
    reference_accepted:17,policy_calibration_status:"pending",policy_rollouts:0};
  payload.calibration.groups=[];
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,localStorage,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(elements.get("status").textContent,/17 accepted \/ 35 attempts \(not model pass rate\)/);
  assert.match(elements.get("status").textContent,/No policy pass rates measured yet/);
  payload.problems.problems[0].initial_images[0].target_annotations[1]={
    role:"destination",target_bbox_status:"not_visible",target_visible_pixels:0,
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,localStorage,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  elements.get("problems").children[0].children[1].children[0].children[0].listeners.click();
  const updated=[];
  function gather(e){updated.push(e);for(const child of e.children || [])gather(child);}
  gather(elements.get("pile-tasks"));
  assert.ok(updated.some(e=>e.textContent.includes("Destination goal marker (cyan) · not visible in this camera · no box")));
  assert.ok(!updated.some(e=>e.stroke==="#00e5ff"));
  assert.ok(updated.some(e=>e.stroke==="#ffdd00"));
  const section=elements.get("problems").children[0];
  section.open=false;section.listeners.toggle();
  elements.get("search").value="first";
  elements.get("search").listeners.input();
  assert.equal(elements.get("problems").children[0].open,false);
  elements.clear();
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,localStorage,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(elements.get("search").value,"first");
  assert.equal(elements.get("bucket").value,"b");
  assert.equal(elements.get("problems").children[0].open,false);
  payload.admitted_piles=[{id:"admission:new",tags:{piece_count:40},initial_images:[]}];
  saved.clear();elements.clear();
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,localStorage,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  const rows=elements.get("populations").children[0].children;
  const pending=rows.find(row=>row.children[0]?.textContent==="40 pieces");
  assert.ok(pending);
  assert.equal(pending.children[1].textContent,"Tasks pending");
  assert.equal(pending.children[3].children[0].textContent,"1 admitted piles · task pool pending");
  const newBucket=elements.get("problems").children.find(e=>e.children[0].textContent==="40 pieces · 1 piles");
  assert.ok(newBucket);
  assert.equal(newBucket.children[1].children[0].children[0].textContent,"40 pieces · admitted pile");
  pending.listeners.click();
  assert.equal(elements.get("bucket").value,"current:40-pieces");
  assert.equal(elements.get("problems").children.length,1);
  payload.manifest.datasets=[{dataset_id:"pool",current_measurement_series_id:"prior",
    measurement_series:[
      {series_id:"prior",label:"Prior measurements",groups:[{problem_id:"first",assistance:.95,attempts:16,successes:8}]},
      {series_id:"new",label:"New measurements",groups:[{problem_id:"first",assistance:.95,attempts:4,successes:0}]},
    ]}];
  saved.clear();elements.clear();
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,localStorage,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  const seriesSelect=elements.get("populations").children[0].children[0];
  const table=elements.get("populations").children[1];
  assert.ok(table.children[1].children[3].children.some(e=>e.textContent==="push: 1/1 problems · 16 rollouts"));
  seriesSelect.value="new";seriesSelect.listeners.change();
  assert.ok(table.children[1].children[3].children.some(e=>e.textContent==="push: 1/1 problems · 4 rollouts"));
  assert.ok(!table.children[1].children[3].children.some(e=>e.textContent.includes("20 rollouts")));
});
