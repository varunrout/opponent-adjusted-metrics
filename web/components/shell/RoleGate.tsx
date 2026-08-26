"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { isRouteAllowed } from "@/lib/nav-config";
import { useRole } from "@/components/shell/RoleProvider";

/**
 * Mirrors the skeleton's `activeHidden` fallback: if the current route
 * becomes inaccessible after a role change, redirect to /overview.
 */
export function RoleGate() {
  const { role, roleResolved } = useRole();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    // Wait for the real role before gating anything.
    //
    // RoleProvider starts at the layout's defaultRole ("guest") and only
    // resolves the true role after /v1/me returns. Without this guard, an
    // admin loading /analysis directly was redirected to /overview on the
    // very first effect run -- while role was still the "guest" default --
    // before the /v1/me round-trip could finish. The admin API calls were
    // therefore never even attempted, which is what made this look like a
    // backend auth failure rather than a client-side race.
    //
    // roleResolved is always eventually set to true by RoleProvider, including
    // on failure or timeout (it settles to "guest"), so this cannot deadlock.
    if (!roleResolved) return;

    if (!isRouteAllowed(pathname, role)) {
      router.replace("/overview");
    }
  }, [role, roleResolved, pathname, router]);

  return null;
}
