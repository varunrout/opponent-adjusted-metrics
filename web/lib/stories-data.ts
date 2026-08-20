export type StoryInfo = {
  category: string;
  headline: string;
};

export const STORIES: StoryInfo[] = [
  { category: "Methodology", headline: "How CxG adjusts for late-game context" },
  { category: "Case study", headline: "A player beating xG by 0.3 per 90 — real or variance" },
  { category: "Release notes", headline: "CxG v1 — what changed and why" },
  // Dev log / behind-the-scenes, per docs/dashboard_content_ideation.md:
  // the Silver _SUCCESS marker ordering defect and its atomic repair is a
  // real production incident from this project's own history, more
  // credible than a polished case study because it's about a mistake and
  // the fix, not a clean result.
  {
    category: "Dev log",
    headline: "The _SUCCESS marker ordering bug — and the atomic repair that fixed it",
  },
];
