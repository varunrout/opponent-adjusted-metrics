import { notFound } from "next/navigation";
import Link from "next/link";
import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";
import { STORIES } from "@/lib/stories-data";

export default function StoryPage({ params }: { params: { slug: string } }) {
  const story = STORIES.find((s) => s.slug === params.slug);
  if (!story) {
    notFound();
  }

  const crumb = [story.date, story.readingTime].filter(Boolean).join(" · ") || story.category;

  return (
    <section>
      <PageHead title={story.headline} crumb={crumb} />

      <Card>
        <span className="text-[10px]" style={{ color: "var(--violet)" }}>
          {story.category}
        </span>

        {story.body && story.body.length > 0 ? (
          <div className="mt-3 flex flex-col gap-3">
            {story.body.map((paragraph, i) => (
              <p key={i} className="text-[13px] text-text2 m-0 leading-relaxed">
                {paragraph}
              </p>
            ))}
          </div>
        ) : (
          <p className="text-[12.5px] text-muted mt-3 mb-0">
            Writeup in progress. This story is on the roadmap but hasn&apos;t been written up yet — check
            back later, or read{" "}
            <Link href="/stories/cxg-v3-honest-comparison" className="text-teal hover:underline">
              the CxG v3 comparison
            </Link>{" "}
            in the meantime.
          </p>
        )}
      </Card>

      <div className="mt-4">
        <Link href="/stories" className="text-[12.5px] text-teal hover:underline">
          ← Back to Stories
        </Link>
      </div>
    </section>
  );
}
