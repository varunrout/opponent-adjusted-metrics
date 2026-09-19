// Pitch markings shared by PitchMap and ShotFreezeFrame — StatsBomb 120x80
// coordinate space, ported from docs/dashboard_layout_skeleton.html's
// .pitch-wrap svg. Extracted so both consumers draw an identical pitch
// instead of duplicating the markup.
export function PitchBackground() {
  return (
    <>
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
    </>
  );
}
