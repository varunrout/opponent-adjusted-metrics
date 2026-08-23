export type Role = "guest" | "viewer" | "admin";

// Per docs/dashboard_design_spec_v2.md §2-3: Explore (dynamic/filterable,
// primary weight), Understand (static/curated, secondary weight), and
// Admin (Analysis, admin-only, its own thing — not part of either).
export type Zone = "explore" | "understand" | "admin";

export type NavItem = {
  href: string;
  label: string;
  roles: Role[];
  secondary?: boolean;
  zone: Zone;
};

// Single source of truth for rendering the nav, gating routes, and (via
// `zone`) which routes get the Explore-only filter sidebar — see
// AppShell.tsx, which derives its Explore route list from this instead of
// maintaining a second copy. Mirrors the `data-target` / `data-roles`
// attributes in docs/dashboard_layout_skeleton.html's #primaryNav, with
// array order now grouped by zone (Explore, then Analysis, then
// Understand) per design_spec_v2.md §3's nav structure table.
export const NAV_ITEMS: NavItem[] = [
  { href: "/matches", label: "Matches", roles: ["guest", "viewer", "admin"], zone: "explore" },
  { href: "/players", label: "Players", roles: ["guest", "viewer", "admin"], zone: "explore" },
  { href: "/teams", label: "Teams", roles: ["guest", "viewer", "admin"], zone: "explore" },
  { href: "/analysis", label: "Analysis", roles: ["admin"], zone: "admin" },
  {
    href: "/overview",
    label: "Overview",
    roles: ["guest", "viewer", "admin"],
    secondary: true,
    zone: "understand",
  },
  {
    href: "/stories",
    label: "Stories",
    roles: ["guest", "viewer", "admin"],
    secondary: true,
    zone: "understand",
  },
  {
    href: "/models",
    label: "Models",
    roles: ["guest", "viewer", "admin"],
    secondary: true,
    zone: "understand",
  },
  {
    href: "/about",
    label: "About",
    roles: ["guest", "viewer", "admin"],
    secondary: true,
    zone: "understand",
  },
];

export function isRouteAllowed(pathname: string, role: Role): boolean {
  const item = NAV_ITEMS.find((i) => pathname === i.href || pathname.startsWith(i.href + "/"));
  if (!item) return true; // unknown routes (e.g. "/") aren't gated here
  return item.roles.includes(role);
}
