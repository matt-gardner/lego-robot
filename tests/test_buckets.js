const {test} = require("node:test");
const assert = require("node:assert/strict");
const {wilson,aggregate} = require("../src/so101/dashboard/buckets.js");
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
    problems:{buckets:[{id:"b",population_total:2,tasks:[{task:"push",population_total:1},{task:"place",population_total:1}]}],
      problems:[{id:"first",bucket:"b",task:"push",initial_images:[{camera:"wrist"}]},
        {id:"second",bucket:"b",task:"place",initial_images:[]}]},
    calibration:{protocol:{rollouts_per_group:4},groups:[{problem_id:"first",assistance:0,attempts:4,successes:4}]},
  };
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/so101/dashboard/buckets.js"),"utf8"),
    {document,fetch:async()=>({ok:true,json:async()=>payload}),console});
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(elements.get("status").textContent,/SCREENING ONLY/);
  assert.match(elements.get("count").textContent,/2 matching problems/);
  const all=[];
  function collect(e){all.push(e);for(const child of e.children || [])collect(child);}
  collect(elements.get("problems"));
  assert.ok(all.some(e=>e.src==="api/buckets/image?problem=first&index=0&snapshot=snapshot"));
  assert.ok(all.some(e=>e.textContent==="Initial scene images not published."));
  assert.ok(all.some(e=>e.textContent==="Unmeasured: no valid rollout counts published."));
});
