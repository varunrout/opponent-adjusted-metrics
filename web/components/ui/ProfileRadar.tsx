// Ported verbatim from docs/dashboard_layout_skeleton.html's profile radar svg.
export function ProfileRadar() {
  return (
    <svg width="100%" height="90" viewBox="0 0 90 90">
      <polygon points="45,8 75,28 63,68 27,68 15,28" fill="none" stroke="var(--border)" strokeWidth="1" />
      <polygon points="45,20 62,32 55,58 35,58 28,32" fill="rgba(139,92,246,0.25)" stroke="#8B5CF6" strokeWidth="1.5" />
    </svg>
  );
}
