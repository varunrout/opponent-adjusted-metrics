"use client";

import type { ChangeEvent } from "react";
import type { Role } from "@/lib/nav-config";
import { useRole } from "@/components/shell/RoleProvider";

export function RoleSwitch() {
  const { role, setRole } = useRole();

  function onChange(e: ChangeEvent<HTMLSelectElement>) {
    setRole(e.target.value as Role);
  }

  return (
    <select
      id="roleSwitch"
      value={role}
      onChange={onChange}
      className="bg-card text-text2 border border-border rounded-lg px-2.5 py-1.5 text-xs"
      aria-label="View as role"
    >
      <option value="guest">View as: Guest</option>
      <option value="viewer">View as: Viewer</option>
      <option value="admin">View as: Admin</option>
    </select>
  );
}
