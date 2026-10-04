import type { MouseEvent } from "react";
/** Spread onto any element for a subtle 3D tilt + moving glare. */
export const tiltProps = {
  onMouseMove: (e: MouseEvent<HTMLElement>) => {
    const el = e.currentTarget, b = el.getBoundingClientRect();
    const x = (e.clientX - b.left) / b.width, y = (e.clientY - b.top) / b.height;
    el.style.transform = `perspective(800px) rotateX(${(0.5 - y) * 6}deg) rotateY(${(x - 0.5) * 8}deg) translateZ(0)`;
    el.style.setProperty("--gx", `${x * 100}%`); el.style.setProperty("--gy", `${y * 100}%`);
  },
  onMouseLeave: (e: MouseEvent<HTMLElement>) => { e.currentTarget.style.transform = ""; },
};
