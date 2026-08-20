"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV_ITEMS } from "@/lib/nav-config";
import { useRole } from "@/components/shell/RoleProvider";

export function PrimaryNav() {
  const { role } = useRole();
  const pathname = usePathname();

  const visible = NAV_ITEMS.filter((item) => item.roles.includes(role));
  const primary = visible.filter((item) => !item.secondary);
  const secondary = visible.filter((item) => item.secondary);

  return (
    <nav className="primary flex flex-1 items-center gap-1 ml-8" aria-label="Primary">
      {primary.map((item) => (
        <NavLink key={item.href} href={item.href} label={item.label} active={pathname.startsWith(item.href)} />
      ))}
      {secondary.length > 0 && <div className="w-px h-[18px] bg-border mx-1.5" />}
      {secondary.map((item) => (
        <NavLink key={item.href} href={item.href} label={item.label} active={pathname.startsWith(item.href)} secondary />
      ))}
    </nav>
  );
}

function NavLink({
  href,
  label,
  active,
  secondary,
}: {
  href: string;
  label: string;
  active: boolean;
  secondary?: boolean;
}) {
  return (
    <Link
      href={href}
      className={[
        "px-3 py-2 rounded-lg text-[13px] cursor-pointer hover:bg-card",
        secondary ? "text-muted" : "text-text2",
        active ? "bg-card text-text" : "",
      ].join(" ")}
    >
      {label}
    </Link>
  );
}
