"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/shell/Sidebar";

// Filters (competition/season/team) only apply in the Explore zone —
// Matches, Players, Teams — per docs/dashboard_design_spec_v2.md section 4.
// Understand-zone pages (Overview, Stories, Models, About) are curated,
// authored content with nothing to filter, and Analysis is its own
// admin-only workbench with its own family/run_id controls, so the
// sidebar shouldn't render on any of those. This is a targeted fix for
// that rendering leak only — the broader Understand/Explore nav-config.ts
// restructure is a separate follow-up, not part of this change.
const EXPLORE_PREFIXES = ["/matches", "/players", "/teams"];

function isExploreRoute(pathname: string): boolean {
  return EXPLORE_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`)
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const showSidebar = isExploreRoute(pathname);

  return (
    <div className="flex" style={{ minHeight: "calc(100vh - 56px)" }}>
      {showSidebar && <Sidebar />}
      <main className="flex-1 px-7 py-6 overflow-hidden">{children}</main>
    </div>
  );
}
