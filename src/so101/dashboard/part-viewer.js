"use strict";

class PartViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.yaw = -0.65;
    this.pitch = 0.55;
    this.zoom = 1;
    this.geometry = null;
    let drag = null;
    canvas.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      drag = {x: event.clientX, y: event.clientY, id: event.pointerId};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      if (!drag || drag.id !== event.pointerId) return;
      this.yaw += (event.clientX - drag.x) * 0.012;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + (event.clientY - drag.y) * 0.012));
      drag.x = event.clientX;
      drag.y = event.clientY;
      this.draw();
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      canvas.addEventListener(name, () => { drag = null; });
    }
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      this.zoom = Math.max(0.5, Math.min(2.5, this.zoom * Math.exp(-event.deltaY * 0.001)));
      this.draw();
    }, {passive: false});
    canvas.addEventListener("keydown", event => {
      const keys = {ArrowLeft: [-0.15, 0], ArrowRight: [0.15, 0],
        ArrowUp: [0, -0.15], ArrowDown: [0, 0.15]};
      if (!keys[event.key]) return;
      event.preventDefault();
      this.yaw += keys[event.key][0];
      this.pitch += keys[event.key][1];
      this.draw();
    });
  }
  setGeometry(geometry) {
    this.geometry = geometry;
    this.yaw = -0.65;
    this.pitch = 0.55;
    this.zoom = 1;
    const vertices = geometry.meshes.flatMap(mesh => mesh.vertices);
    const lower = [Infinity, Infinity, Infinity], upper = [-Infinity, -Infinity, -Infinity];
    for (const vertex of vertices) for (let axis = 0; axis < 3; axis++) {
      lower[axis] = Math.min(lower[axis], vertex[axis]);
      upper[axis] = Math.max(upper[axis], vertex[axis]);
    }
    this.center = lower.map((value, axis) => (value + upper[axis]) / 2);
    this.radius = 0;
    for (const vertex of vertices) {
      this.radius = Math.max(this.radius, Math.hypot(...vertex.map((value, axis) => value - this.center[axis])));
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) throw new Error("Degenerate part geometry");
    this.draw();
  }
  draw() {
    if (!this.geometry) return;
    const canvas = this.canvas, ctx = canvas.getContext("2d");
    const width = canvas.width, height = canvas.height;
    ctx.fillStyle = "#eef2f6";
    ctx.fillRect(0, 0, width, height);
    const triangles = [];
    for (const mesh of this.geometry.meshes) {
      const points = mesh.vertices.map(vertex => {
        const [x, y, z] = vertex.map((value, axis) => (value - this.center[axis]) / this.radius);
        const horizontal = Math.cos(this.yaw) * x - Math.sin(this.yaw) * y;
        const depth = Math.sin(this.yaw) * x + Math.cos(this.yaw) * y;
        const vertical = Math.cos(this.pitch) * z - Math.sin(this.pitch) * depth;
        const rotatedDepth = Math.cos(this.pitch) * depth + Math.sin(this.pitch) * z;
        const scale = Math.min(width, height) * 0.39 * this.zoom;
        return [width / 2 + horizontal * scale, height / 2 - vertical * scale, rotatedDepth];
      });
      for (const face of mesh.faces) {
        const vertices = face.map(index => points[index]);
        const [a, b, c] = face.map(index => mesh.vertices[index]);
        const u = b.map((value, axis) => value - a[axis]), v = c.map((value, axis) => value - a[axis]);
        const normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        const length = Math.hypot(...normal);
        const shade = length ? 0.4 + 0.6 * Math.abs((normal[0] * 0.35 - normal[1] * 0.5 + normal[2] * 0.79) / length) : 0.4;
        triangles.push({vertices, depth: vertices.reduce((sum, vertex) => sum + vertex[2], 0) / 3,
          color: mesh.color, shade});
      }
    }
    triangles.sort((a, b) => a.depth - b.depth);
    for (const triangle of triangles) {
      const [a, b, c] = triangle.vertices;
      const color = `rgb(${triangle.color.slice(0, 3).map(value => Math.round(value * triangle.shade * 255)).join(",")})`;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.lineTo(c[0], c[1]);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.5;
      ctx.fill();
      ctx.stroke();
    }
  }
}
