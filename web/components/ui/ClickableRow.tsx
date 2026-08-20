"use client";

import { useRouter } from "next/navigation";
import type { KeyboardEvent, ReactNode } from "react";

/**
 * A list row that navigates to `href` on click, without being an <a> itself.
 * Used where a row needs nested links (e.g. a team name linking into /teams/[id])
 * — a real Next <Link> can't wrap another <Link> without producing invalid,
 * broken-click nested <a> markup.
 */
export function ClickableRow({
  href,
  className,
  children,
}: {
  href: string;
  className?: string;
  children: ReactNode;
}) {
  const router = useRouter();

  function navigate() {
    router.push(href);
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      navigate();
    }
  }

  return (
    <div role="link" tabIndex={0} onClick={navigate} onKeyDown={onKeyDown} className={className}>
      {children}
    </div>
  );
}
