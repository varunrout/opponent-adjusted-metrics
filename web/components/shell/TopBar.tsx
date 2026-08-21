"use client";

import Link from "next/link";
import { signOut } from "firebase/auth";
import { PrimaryNav } from "@/components/shell/PrimaryNav";
import { RoleSwitch } from "@/components/shell/RoleSwitch";
import { useRole } from "@/components/shell/RoleProvider";
import { auth } from "@/lib/firebase";

export function TopBar() {
  const { role, user } = useRole();

  function handleAvatarClick() {
    if (auth && user) {
      signOut(auth);
    }
  }

  const initials = user?.email ? user.email.slice(0, 2).toUpperCase() : "VR";

  return (
    <div className="h-14 flex items-center justify-between px-5 bg-surface border-b border-border sticky top-0 z-10">
      <div className="flex items-center gap-2.5">
        <div className="w-[26px] h-[26px] rounded-md bg-teal flex items-center justify-center font-bold text-[13px]" style={{ color: "#04231f" }}>
          OA
        </div>
        <div className="text-[13px] text-text2">Opponent-Adjusted Metrics</div>
      </div>

      <PrimaryNav />

      <div className="flex items-center gap-3">
        {process.env.NODE_ENV !== "production" && <RoleSwitch />}
        {role === "guest" && !user ? (
          <Link
            href="/login"
            className="flex items-center gap-2 text-[12px] text-text2 hover:text-text hover:underline"
            aria-label="Sign in"
            title="Sign in"
          >
            <span>Sign in</span>
            <span
              className="w-7 h-7 rounded-full bg-violet flex items-center justify-center text-[11px] font-semibold"
              style={{ color: "#1c1030" }}
            >
              VR
            </span>
          </Link>
        ) : (
          <button
            type="button"
            onClick={handleAvatarClick}
            className="w-7 h-7 rounded-full bg-violet flex items-center justify-center text-[11px] font-semibold"
            style={{ color: "#1c1030" }}
            aria-label="Sign out"
            title="Sign out"
          >
            {initials}
          </button>
        )}
      </div>
    </div>
  );
}
