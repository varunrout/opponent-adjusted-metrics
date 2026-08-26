export type StoryInfo = {
  slug: string;
  category: string;
  headline: string;
  // Per docs/dashboard_design_spec_v2.md §10's component inventory
  // ("Story card | Category tag, headline, takeaway, thumbnail"): a
  // one-line summary of the finding. Optional so existing entries don't
  // need a retrofitted takeaway invented for them.
  takeaway?: string;
};

export const STORIES: StoryInfo[] = [
  {
    slug: "cxg-late-game-context",
    category: "Methodology",
    headline: "How CxG adjusts for late-game context",
  },
  {
    slug: "beating-xg-by-0-3-per-90",
    category: "Case study",
    headline: "A player beating xG by 0.3 per 90 — real or variance",
  },
  { slug: "cxg-v1-release-notes", category: "Release notes", headline: "CxG v1 — what changed and why" },
  // Dev log / behind-the-scenes, per docs/dashboard_content_ideation.md:
  // the Silver _SUCCESS marker ordering defect and its atomic repair is a
  // real production incident from this project's own history, more
  // credible than a polished case study because it's about a mistake and
  // the fix, not a clean result.
  {
    slug: "success-marker-ordering-bug",
    category: "Dev log",
    headline: "The _SUCCESS marker ordering bug — and the atomic repair that fixed it",
  },
  // CxG v3's honest comparison against the StatsBomb baseline, per
  // docs/dashboard_design_spec_v2.md §4a/§11 (Hard gate 1's reframing —
  // "must be honestly compared," not "must beat the baseline"). This is
  // the first (and, for now, only) story with a full written article —
  // see app/stories/[slug]/page.tsx and ARTICLE_BODIES below.
  {
    slug: "cxg-v3-honest-comparison",
    category: "Methodology",
    headline: "CxG v3 — an honest comparison against StatsBomb xG",
    takeaway:
      "CxG v3 improved over every prior version but still trails the StatsBomb baseline — and that's a legitimate, disclosed result, not a hidden one.",
  },
];
