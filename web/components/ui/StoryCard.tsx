import Link from "next/link";
import type { StoryInfo } from "@/lib/stories-data";

export function StoryCard({ story }: { story: StoryInfo }) {
  return (
    <Link
      href={`/stories/${story.slug}`}
      className="block bg-card border border-border rounded p-3.5 hover:bg-card-hi transition-colors"
    >
      <span className="text-[10px]" style={{ color: "var(--violet)" }}>
        {story.category}
      </span>
      <p className="text-[13px] mt-1.5 mb-0">{story.headline}</p>
      {story.takeaway && <p className="text-[11.5px] text-muted mt-1.5 mb-0">{story.takeaway}</p>}
    </Link>
  );
}
