/** Lightweight inline icons for training UI (no lucide dependency). */

import type { ReactNode } from "react";

type IconProps = { size?: number; className?: string };

function Svg({ size = 16, className, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {children}
    </svg>
  );
}

export function IconPlay(props: IconProps) {
  return <Svg {...props}><path d="M8 5.5v13l11-6.5L8 5.5Z" fill="currentColor" stroke="none" /></Svg>;
}
export function IconStop(props: IconProps) {
  return <Svg {...props}><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none" /></Svg>;
}
export function IconList(props: IconProps) {
  return <Svg {...props}><path d="M8 6h12M8 12h12M8 18h12" /><circle cx="4" cy="6" r="1" fill="currentColor" stroke="none" /><circle cx="4" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="4" cy="18" r="1" fill="currentColor" stroke="none" /></Svg>;
}
export function IconSave(props: IconProps) {
  return <Svg {...props}><path d="M5 5h11l3 3v11H5V5Z" /><path d="M8 5v5h8V5M8 19v-6h8v6" /></Svg>;
}
export function IconAlert(props: IconProps) {
  return <Svg {...props}><path d="M12 3.5 21 19H3L12 3.5Z" /><path d="M12 10v4M12 17h.01" /></Svg>;
}
export function IconClock(props: IconProps) {
  return <Svg {...props}><circle cx="12" cy="12" r="8" /><path d="M12 8v5l3 2" /></Svg>;
}
export function IconCpu(props: IconProps) {
  return <Svg {...props}><rect x="7" y="7" width="10" height="10" rx="1.5" /><path d="M10 7V4M14 7V4M10 20v-3M14 20v-3M7 10H4M7 14H4M20 10h-3M20 14h-3" /></Svg>;
}
export function IconSpark(props: IconProps) {
  return <Svg {...props}><path d="M12 3l1.4 5.2L18 9.5l-4.6 1.3L12 16l-1.4-5.2L6 9.5l4.6-1.3L12 3Z" /><path d="M19 14l.7 2.3L22 17l-2.3.7L19 20l-.7-2.3L16 17l2.3-.7L19 14Z" /></Svg>;
}
export function IconChevron(props: IconProps) {
  return <Svg {...props}><path d="m6 9 6 6 6-6" /></Svg>;
}
export function IconSearch(props: IconProps) {
  return <Svg {...props}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4 4" /></Svg>;
}
export function IconPlus(props: IconProps) {
  return <Svg {...props} size={props.size ?? 28}><path d="M12 5v14M5 12h14" /></Svg>;
}
export function IconBox(props: IconProps) {
  return <Svg {...props}><path d="M4 8.5 12 4l8 4.5v7L12 20l-8-4.5v-7Z" /><path d="M12 12v8M4 8.5l8 3.5 8-3.5" /></Svg>;
}
export function IconShapes(props: IconProps) {
  return <Svg {...props}><path d="M4 16.5 8 7l4 9.5H4Z" /><circle cx="16.5" cy="14.5" r="4.5" /></Svg>;
}
export function IconScan(props: IconProps) {
  return <Svg {...props}><path d="M4 8V5h3M17 5h3v3M20 16v3h-3M7 19H4v-3" /><rect x="8" y="8" width="8" height="8" rx="1" transform="rotate(15 12 12)" /></Svg>;
}
export function IconGrid(props: IconProps) {
  return <Svg {...props}><rect x="4" y="4" width="7" height="7" rx="1" /><rect x="13" y="4" width="7" height="7" rx="1" /><rect x="4" y="13" width="7" height="7" rx="1" /><rect x="13" y="13" width="7" height="7" rx="1" /></Svg>;
}
export function IconBack(props: IconProps) {
  return <Svg {...props}><path d="M15 6 9 12l6 6" /></Svg>;
}
