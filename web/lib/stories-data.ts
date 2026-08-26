export type StoryInfo = {
  slug: string;
  category: string;
  headline: string;
  // Per docs/dashboard_design_spec_v2.md §10's component inventory
  // ("Story card | Category tag, headline, takeaway, thumbnail"): a
  // one-line summary of the finding. Optional so existing entries don't
  // need a retrofitted takeaway invented for them.
  takeaway?: string;
  // Authoring format decision (v3 §12.2): a `body` field of plain paragraph
  // strings in this data file, not MDX. MDX (@next/mdx) would read better
  // for long articles and is the right call if this grows past a handful of
  // stories, but it's a new dependency + build-config change for a single
  // article today. Paragraphs render 1:1, in order; keep them short.
  date?: string;
  readingTime?: string;
  body?: string[];
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
    date: "2026-08-26",
    readingTime: "4 min",
    body: [
      "CxG is trained on two separate tracks: an event-wide track (`cxg_event`, 8 features, no 360 tracking data required) and CxG+ (24 features, using StatsBomb's 360 freeze-frame data where available). Both were re-trained for v3 and evaluated on a fixed test split against the StatsBomb xG baseline on the same shots. On every one of six metrics, across both tracks, CxG v3 trails StatsBomb.",
      "Event-wide track (cxg_event): log_loss 0.3003 vs StatsBomb's 0.2597. Brier score 0.0852 vs 0.0718. AUC 0.7148 vs 0.7972.",
      "CxG+ track: log_loss 0.2555 vs StatsBomb's 0.2430. Brier score 0.0713 vs 0.0665. AUC 0.8313 vs 0.8476.",
      "That is six out of six metrics losing to a well-established industry baseline. It is disclosed here, with the numbers intact, rather than reframed or buried in an appendix, because a model that loses honestly is more useful — to a reader trying to judge this work, and to future iterations of the model itself — than a model whose losses are hidden.",
      "Two caveats matter for reading this table correctly. First, feature-pool asymmetry: the event-wide track uses 8 features and CxG+ uses 24. The two tracks are not a like-for-like test of whether 360 tracking data helps — some of the gap between them reflects feature count alone, not just data richness. Second, one of CxG+'s features, `zone_displacement`, shows an unexplained bimodal distribution in its values. That's flagged here as an open question, not a resolved one — it should not be presented as a clean, well-understood feature until the cause is found.",
      "What it would take to close the gap: more feature engineering on both tracks, resolving the `zone_displacement` bimodality before trusting what CxG+ is actually learning from it, and — separately from modelling work — building an honest widened-coverage view of the existing predictions (the model was also evaluated on far more shots than the dashboard currently shows; see the Models page for status). None of that has happened yet. This page states where things stand today, not where they're expected to end up.",
    ],
  },
];
