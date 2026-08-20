"use client";

import Link from "next/link";
import type { MouseEvent } from "react";

/**
 * Cross-linking helpers: team/player names → their detail pages. Safe to
 * nest inside a ClickableRow (stops the click from also firing the row's
 * own navigation) or to use standalone. Falls back to plain text when the
 * id isn't known, rather than linking to a broken/empty page.
 */
export function TeamLink({
  teamId,
  name,
  className,
}: {
  teamId: number | null;
  name: string;
  className?: string;
}) {
  if (teamId == null) {
    return <span className={className}>{name}</span>;
  }
  return (
    <Link
      href={`/teams/${teamId}`}
      className={className ? `${className} hover:underline` : "hover:underline"}
      onClick={(e: MouseEvent) => e.stopPropagation()}
    >
      {name}
    </Link>
  );
}

export function PlayerLink({
  playerId,
  name,
  className,
}: {
  playerId: number | null;
  name: string;
  className?: string;
}) {
  if (playerId == null) {
    return <span className={className}>{name}</span>;
  }
  return (
    <Link
      href={`/players/${playerId}`}
      className={className ? `${className} hover:underline` : "hover:underline"}
      onClick={(e: MouseEvent) => e.stopPropagation()}
    >
      {name}
    </Link>
  );
}
