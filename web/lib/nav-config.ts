export type Role = "guest" | "viewer" | "admin";

export type NavItem = {
  href: string;
  label: string;
  roles: Role[];
  secondary?: boolean;
};

// Single source of truth for both rendering the nav and gating routes.
// Mirrors the `data-target` / `data-roles` attributes in
// docs/dashboard_layout_skeleton.html's #primaryNav.
export const NAV_ITEMS: NavItem[] = [
  { href: "/overview", label: "Overview", roles: ["guest", "viewer", "admin"] },
  { href: "/matches", label: "Matches", roles: ["guest", "viewer", "admin"] },
  { href: "/players", label: "Players", roles: ["guest", "viewer", "admin"] },
  { href: "/teams", label: "Teams", roles: ["guest", "viewer", "admin"] },
  { href: "/analysis", label: "Analysis", roles: ["admin"] },
  { href: "/models", label: "Models", roles: ["guest", "viewer", "admin"], secondary: true },
  { href: "/stories", label: "Stories", roles: ["guest", "viewer", "admin"], secondary: true },
  { href: "/about", label: "About", roles: ["guest", "viewer", "admin"], secondary: true },
];

export function isRouteAllowed(pathname: string, role: Role): boolean {
  const item = NAV_ITEMS.find((i) => pathname === i.href || pathname.startsWith(i.href + "/"));
  if (!item) return true; // unknown routes (e.g. "/") aren't gated here
  return item.roles.includes(role);
}
