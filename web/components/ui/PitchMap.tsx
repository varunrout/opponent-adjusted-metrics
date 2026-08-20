// Ported verbatim from docs/dashboard_layout_skeleton.html's .pitch-wrap svg
// (StatsBomb 120x80 coordinate space per the design spec's PitchMap primitive).
export function PitchMap() {
  return (
    <svg viewBox="0 0 120 80" className="w-full h-auto block rounded-md">
      <rect x="0" y="0" width="120" height="80" fill="#123b25" />
      <g stroke="rgba(255,255,255,0.35)" strokeWidth="0.4" fill="none">
        <rect x="0.4" y="0.4" width="119.2" height="79.2" />
        <line x1="60" y1="0" x2="60" y2="80" />
        <circle cx="60" cy="40" r="9.15" />
        <rect x="0" y="18" width="18" height="44" />
        <rect x="102" y="18" width="18" height="44" />
        <rect x="0" y="30" width="6" height="20" />
        <rect x="114" y="30" width="6" height="20" />
      </g>
      <circle cx="112" cy="39" r="4.4" fill="#14B8A6" fillOpacity="0.9" stroke="#0b0e12" strokeWidth="0.5" />
      <circle cx="99" cy="47" r="2.2" fill="#14B8A6" fillOpacity="0.2" stroke="#14B8A6" strokeWidth="0.5" />
      <circle cx="104" cy="30" r="1.9" fill="#14B8A6" fillOpacity="0.2" stroke="#14B8A6" strokeWidth="0.5" />
      <circle cx="9" cy="43" r="2.6" fill="#F97316" fillOpacity="0.2" stroke="#F97316" strokeWidth="0.5" />
      <circle cx="7" cy="40" r="4.9" fill="#F97316" fillOpacity="0.9" stroke="#0b0e12" strokeWidth="0.5" />
    </svg>
  );
}
