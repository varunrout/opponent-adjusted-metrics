"use client";

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { Role } from "@/lib/nav-config";

type RoleContextValue = {
  role: Role;
  setRole: (role: Role) => void;
};

const RoleContext = createContext<RoleContextValue | undefined>(undefined);

export function RoleProvider({
  children,
  defaultRole = "admin",
}: {
  children: ReactNode;
  defaultRole?: Role;
}) {
  const [role, setRole] = useState<Role>(defaultRole);
  const value = useMemo(() => ({ role, setRole }), [role]);
  return <RoleContext.Provider value={value}>{children}</RoleContext.Provider>;
}

export function useRole(): RoleContextValue {
  const ctx = useContext(RoleContext);
  if (!ctx) {
    throw new Error("useRole must be used within a RoleProvider");
  }
  return ctx;
}
