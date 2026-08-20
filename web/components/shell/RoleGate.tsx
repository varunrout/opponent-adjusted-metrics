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
  const { role } = useRole();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (!isRouteAllowed(pathname, role)) {
      router.replace("/overview");
    }
  }, [role, pathname, router]);

  return null;
}
