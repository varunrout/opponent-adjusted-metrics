import { PageHead } from "@/components/ui/PageHead";
import { StoryCard } from "@/components/ui/StoryCard";
import { STORIES } from "@/lib/stories-data";

export default function StoriesPage() {
  return (
    <section>
      <PageHead title="Stories" crumb="Writeups" />
      <div className="grid gap-3.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
        {STORIES.map((story) => (
          <StoryCard key={story.headline} story={story} />
        ))}
      </div>
    </section>
  );
}
