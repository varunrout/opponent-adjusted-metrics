export type StoryInfo = {
  category: string;
  headline: string;
  // Per docs/dashboard_design_spec_v2.md §10's component inventory
  // ("Story card | Category tag, headline, takeaway, thumbnail"): a
  // one-line summary of the finding. Optional so existing entries don't
  // need a retrofitted takeaway invented for them.
  takeaway?: string;
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
  // CxG v3's honest comparison against the StatsBomb baseline, per
  // docs/dashboard_design_spec_v2.md §4a/§11 (Hard gate 1's reframing —
  // "must be honestly compared," not "must beat the baseline"). Numbers
  // live on the Models cards and Analysis tab; this stays qualitative.
  {
    category: "Methodology",
    headline: "CxG v3 — an honest comparison against StatsBomb xG",
    takeaway:
      "CxG v3 improved over every prior version but still trails the StatsBomb baseline — and that's a legitimate, disclosed result, not a hidden one.",
  },
];
