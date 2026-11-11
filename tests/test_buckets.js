const {test} = require("node:test");
const assert = require("node:assert/strict");
const {wilson,aggregate,groupPiles} = require("../src/so101/dashboard/buckets.js");
const vm = require("node:vm");
const fs = require("node:fs");
test("four successes do not imply certain success or calibration",()=>{
  const ci=wilson(4,4);
  assert.ok(ci[0]<0.52 && ci[0]>0.50);
  assert.ok(ci[1]<=1);
  assert.equal(wilson(0,0),null);
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
  }
  const elements=new Map();
  const document={
    getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
    createElement(){return new Element();},
    createElementNS(){return new Element();},
  };
  const payload={
    manifest:{snapshot_id:"snapshot"},
    problems:{buckets:[{id:"b",population_total:2,tasks:[{task:"push",population_total:1},{task:"place",population_total:1}]},
        {id:"corpus:old",population_total:1,tasks:[{task:"grasp",population_total:1}]}],
      problems:[{id:"first",bucket:"b",task:"push",initial_images:[{camera:"wrist"}]},
        {id:"second",bucket:"b",task:"place",initial_images:[]},
        {id:"old",bucket:"corpus:old",task:"grasp",source:{kind:"legacy"},initial_images:[]}]},
    calibration:{protocol:{rollouts_per_group:4},groups:[{problem_id:"first",assistance:0,attempts:4,successes:4}]},
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(elements.get("status").textContent,/SCREENING ONLY/);
  assert.match(elements.get("count").textContent,/2 matching problems/);
  assert.equal(elements.get("problems").children.length,1);
  assert.equal(elements.get("problems").children[0].children[1].className,"pile-group");
  assert.equal(elements.get("problems").children[0].children[1].children[1].className,"problem");
  const all=[];
  function collect(e){all.push(e);for(const child of e.children || [])collect(child);}
  collect(elements.get("problems"));
  assert.ok(all.some(e=>e.src==="api/buckets/image?problem=first&index=0&snapshot=snapshot"));
  assert.ok(all.some(e=>e.textContent==="Initial scene images not published."));
  assert.ok(all.some(e=>e.textContent==="Unmeasured: no valid rollout counts published."));
  const overview=elements.get("populations").children[0];
  assert.equal(overview.children.length,3);
  assert.equal(elements.get("bucket").children.length,1);
  assert.equal(elements.get("task").children.length,2);
  assert.ok(!all.some(e=>e.textContent.includes("corpus:old")));
  const measured=overview.children.find(row=>row.children[1]?.textContent==="push");
  const unmeasured=overview.children.find(row=>row.children[1]?.textContent==="place");
  assert.equal(measured.children[3].textContent,"0");
  assert.equal(measured.children[4].textContent,"100.0% (4/4)");
  assert.equal(unmeasured.children[3].textContent,"Unmeasured");
  measured.listeners.click();
  assert.equal(elements.get("bucket").value,"b");
  assert.match(elements.get("count").textContent,/1 matching problems/);
});
