"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { onAuthStateChanged, type User } from "firebase/auth";
import type { Role } from "@/lib/nav-config";
import { auth } from "@/lib/firebase";
import { getMe } from "@/lib/api";

type RoleContextValue = {
  role: Role;
  setRole: (role: Role) => void;
  // True once the initial auth-derived role resolution has completed (or
  // been skipped because auth isn't configured). Exposed for callers that
  // want to avoid flashing a "guest" state before the real role is known;
  // not required reading for existing consumers.
  roleResolved: boolean;
  user: User | null;
};

const RoleContext = createContext<RoleContextValue | undefined>(undefined);

// If GET /v1/me hangs (slow/unresponsive backend), roleResolved must still
// settle to true so any page gating on it (e.g. /analysis) doesn't spin on
// its loading skeleton forever with zero requests firing. This races the
// real call against a fixed timeout and falls back to the existing guest
// path on either a rejection or a timeout.
const ME_TIMEOUT_MS = 9000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Timed out after ${ms}ms`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

export function RoleProvider({
  children,
  defaultRole = "admin",
}: {
  children: ReactNode;
  defaultRole?: Role;
}) {
  const [role, setRole] = useState<Role>(defaultRole);
  const [roleResolved, setRoleResolved] = useState(false);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // No configured Firebase auth (e.g. env vars unset in this
    // environment) — settle on guest and never attempt a network call.
    if (!auth) {
      setUser(null);
      setRole("guest");
      setRoleResolved(true);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);
      if (!firebaseUser) {
        setRole("guest");
        setRoleResolved(true);
        return;
      }
      try {
        const idToken = await firebaseUser.getIdToken();
        const me = await withTimeout(getMe(idToken), ME_TIMEOUT_MS);
        setRole(me.role);
      } catch {
        // Never let a role-resolution failure (network error, 401, a
        // hung/timed-out backend, etc.) crash the app or leave
        // roleResolved stuck at false — fall back to guest.
        setRole("guest");
      } finally {
        setRoleResolved(true);
      }
    });

    return () => unsubscribe();
  }, []);

  const value = useMemo(
    () => ({ role, setRole, roleResolved, user }),
    [role, roleResolved, user]
  );
  return <RoleContext.Provider value={value}>{children}</RoleContext.Provider>;
}

export function useRole(): RoleContextValue {
  const ctx = useContext(RoleContext);
  if (!ctx) {
    throw new Error("useRole must be used within a RoleProvider");
  }
  return ctx;
}
