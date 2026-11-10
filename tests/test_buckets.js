const {test} = require("node:test");
const assert = require("node:assert/strict");
const {wilson,aggregate} = require("../src/so101/dashboard/buckets.js");
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
