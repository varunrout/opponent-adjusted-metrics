import type { ReactNode } from "react";

/**
 * Shared shell for every embedded story figure — same bordered-card
 * language as the rest of the dashboard (`Card`), with a title and an
 * optional caption slot so individual figures stay pure data/markup.
 */
export function StoryFigureCard({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: ReactNode;
}) {
  return (
    <div className="bg-card border border-border rounded p-3.5 my-1" data-testid="story-figure">
      <h3 className="text-[11.5px] font-medium text-text2 m-0 mb-3">{title}</h3>
      {children}
      {caption && <p className="text-[11px] text-muted mt-3 mb-0 leading-relaxed">{caption}</p>}
    </div>
  );
}
