"use client";
import { useEffect, useRef } from "react";

/** Dependency-free 3D point cloud (perspective-projected sphere of "evidence nodes").
 *  idle: slow drift + mouse parallax. think: nodes contract, spin faster and pulse. */
export default function Constellation({ mode = "idle", height = 240, className = "" }: { mode?: "idle" | "think"; height?: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;

  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const N = 110;
    const pts = Array.from({ length: N }, (_, i) => { // fibonacci sphere
      const y = 1 - (i / (N - 1)) * 2, r = Math.sqrt(1 - y * y), t = i * 2.399963;
      return { x: Math.cos(t) * r, y, z: Math.sin(t) * r };
    });
    let w = 0, h = 0, rot = 0, spin = 0.002, k = 1, mx = 0, my = 0, raf = 0, t0 = performance.now();
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim() || "#6ea8fe";
    const resize = () => {
      const d = Math.min(window.devicePixelRatio || 1, 2), b = cv.getBoundingClientRect();
      w = b.width; h = b.height; cv.width = w * d; cv.height = h * d; ctx.setTransform(d, 0, 0, d, 0, 0);
    };
    const move = (e: MouseEvent) => { const b = cv.getBoundingClientRect(); mx = ((e.clientX - b.left) / b.width - 0.5); my = ((e.clientY - b.top) / b.height - 0.5); };
    const frame = (now: number) => {
      const think = modeRef.current === "think", s = (now - t0) / 1000;
      spin += ((think ? 0.018 : 0.002) - spin) * 0.05;
      k += ((think ? 0.55 + 0.06 * Math.sin(s * 5) : 1) - k) * 0.06;
      rot += spin;
      const R = Math.min(w, h) * 0.38 * k, cx = w / 2, cy = h / 2, tilt = my * 0.6, ry = rot + mx * 0.8;
      const P = pts.map((p) => {
        const x1 = p.x * Math.cos(ry) - p.z * Math.sin(ry), z1 = p.x * Math.sin(ry) + p.z * Math.cos(ry);
        const y2 = p.y * Math.cos(tilt) - z1 * Math.sin(tilt), z2 = p.y * Math.sin(tilt) + z1 * Math.cos(tilt);
        const f = 1 / (1.8 - z2 * 0.6);
        return { x: cx + x1 * R * f * 1.8, y: cy + y2 * R * f * 1.8, z: z2, f };
      });
      ctx.clearRect(0, 0, w, h);
      ctx.lineWidth = 0.6; ctx.strokeStyle = accent;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j += 3) {
        const dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d = dx * dx + dy * dy, lim = (R * 0.42) ** 2;
        if (d < lim) { ctx.globalAlpha = (1 - d / lim) * 0.28 * ((P[i].z + P[j].z + 2) / 4 + 0.2); ctx.beginPath(); ctx.moveTo(P[i].x, P[i].y); ctx.lineTo(P[j].x, P[j].y); ctx.stroke(); }
      }
      ctx.fillStyle = accent;
      P.forEach((p, i) => {
        const pulse = think ? 0.5 + 0.5 * Math.sin(s * 6 - i * 0.3) : 0.6;
        ctx.globalAlpha = Math.max(0.12, (p.z + 1) / 2) * (0.5 + 0.5 * pulse);
        ctx.beginPath(); ctx.arc(p.x, p.y, 1 + 1.8 * p.f * (0.6 + pulse * 0.6), 0, 6.283); ctx.fill();
      });
      ctx.globalAlpha = 1;
      if (!reduce) raf = requestAnimationFrame(frame);
    };
    resize(); raf = requestAnimationFrame(frame);
    window.addEventListener("resize", resize); window.addEventListener("mousemove", move);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); window.removeEventListener("mousemove", move); };
  }, []);

  return <canvas ref={ref} className={className} style={{ width: "100%", height, display: "block" }} aria-hidden="true" />;
}
