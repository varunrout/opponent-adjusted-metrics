import type { StoryInfo } from "@/lib/stories-data";

export function StoryCard({ story }: { story: StoryInfo }) {
  return (
    <div className="bg-card border border-border rounded p-3.5">
      <span className="text-[10px]" style={{ color: "var(--violet)" }}>
        {story.category}
      </span>
      <p className="text-[13px] mt-1.5 mb-0">{story.headline}</p>
      {story.takeaway && <p className="text-[11.5px] text-muted mt-1.5 mb-0">{story.takeaway}</p>}
    </div>
  );
}
