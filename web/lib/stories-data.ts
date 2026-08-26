export type StoryInfo = {
  slug: string;
  category: string;
  headline: string;
  // Per docs/dashboard_design_spec_v2.md §10's component inventory
  // ("Story card | Category tag, headline, takeaway, thumbnail"): a
  // one-line summary of the finding.
  takeaway?: string;
  // Authoring format decision (v3 §12.2): a `body` field of plain paragraph
  // strings in this data file, not MDX. MDX (@next/mdx) would read better
  // for long articles and is the right call if this grows past a handful of
  // stories, but it's a new dependency + build-config change today.
  // Paragraphs render 1:1, in order, as plain text: no markdown, so avoid
  // backticks and tables and write figures into the prose instead.
  date?: string;
  readingTime?: string;
  body?: string[];
};

export const STORIES: StoryInfo[] = [
  {
    slug: "cxg-v3-honest-comparison",
    category: "Methodology",
    headline: "CxG v3 against StatsBomb xG: an honest comparison",
    takeaway:
      "CxG v3 beat every version of itself and still lost to StatsBomb on all six metrics. I'm publishing the numbers anyway.",
    date: "2026-08-26",
    readingTime: "4 min",
    body: [
      "CxG runs on two tracks. The event-wide track uses 8 features and needs no tracking data. CxG+ uses 24 and leans on StatsBomb's 360 freeze frames where they exist. I retrained both for v3 and scored them against StatsBomb's own xG on the same held-out shots.",
      "It lost. Six metrics, two tracks, every one of them.",
      "Event-wide: log loss 0.3003 against StatsBomb's 0.2597. Brier 0.0852 against 0.0718. AUC 0.7148 against 0.7972.",
      "CxG+: log loss 0.2555 against 0.2430. Brier 0.0713 against 0.0665. AUC 0.8313 against 0.8476.",
      "I could have buried this in an appendix or picked the one framing where it looks closer. I'd rather you see it. StatsBomb's model has years of proprietary data and a team behind it. Mine has a fixed open dataset and me. Losing to it is the expected outcome, and pretending otherwise would make everything else on this site harder to trust.",
      "Two caveats, because the table is easy to misread. The tracks aren't a clean test of whether 360 data helps: one has 8 features and the other has 24, so part of that gap is just feature count. And one CxG+ feature, zone displacement, has a bimodal distribution I can't explain yet. It's in the model, it's flagged, and I'm not going to pretend I know what it's doing.",
      "To close the gap I'd need better feature engineering on both tracks and an answer on zone displacement before I trust what CxG+ is learning from it. Neither has happened. This is where the model actually stands, not where I expect it to end up.",
    ],
  },
  {
    slug: "late-game-features-that-failed",
    category: "Methodology",
    headline: "I built late-game context features and they failed",
    takeaway:
      "Scoreline and time-remaining features felt obviously right. Fifteen out of fifteen interaction pairs failed on held-out data.",
    date: "2026-08-26",
    readingTime: "3 min",
    body: [
      "A shot at 2-0 down in the 88th minute isn't the same shot as one at 0-0 in the 20th. Defences sit differently, keepers commit differently, and the shooter is under different pressure. That felt obvious enough to build.",
      "So I built it. Seven features: score difference, game state, match minute, regulation time remaining, manpower difference, and two flags for being ahead or behind late. Late-game trailing fires past the 75th minute when you're behind. Late-game leading is the mirror image.",
      "Then I tested them properly and they all died.",
      "Match minute went first, in univariate screening. Six candidates cleared the sign-stability and minimum-correlation thresholds on the event-wide track, and I trimmed to five. Match minute wasn't one of them.",
      "The rest got a dedicated bivariate test: three defensive-index features crossed with five match-context features, fifteen pairs in total. None of the fifteen validated on the held-out split.",
      "Three looked good on training data alone. Goalkeeper index crossed with manpower difference at p=0.0145. Nearest-defender index crossed with manpower difference at p=0.0292. Nearest-defender index crossed with late-game leading at p=0.0327. All three failed held-out confirmation, which is exactly what a held-out split is for.",
      "So neither shipped model knows what minute it is. Event-wide has 8 features, CxG+ has 24, and not one of them is about time or scoreline.",
      "I've left the features in the contract and the EDA charts in the appendix. They cost nothing to keep and the next person to have this idea, including future me, should be able to see it was already tried.",
      "If I revisit it, I'd question the grain rather than the intuition. Testing late-game effects as pairwise interactions on a few thousand shots may just be asking too much of the sample. The idea might still be right. This particular test of it wasn't.",
    ],
  },
  {
    slug: "everything-was-3x-too-big",
    category: "Dev log",
    headline: "Every number on the site was 3x too big and nothing looked wrong",
    takeaway:
      "Shot counts, goals and xG were all inflated threefold on live pages. The bug survived because it was perfectly consistent.",
    date: "2026-08-26",
    readingTime: "3 min",
    body: [
      "For a while, every number a visitor saw on Matches, Players and Teams was three times too big. Not occasionally. Every number, every page, every visit.",
      "The cause is boring, which is part of the point. My core tables keep one full copy of every row per schema version, and there are three of them: v1, v1_1 and v1_2. Three copies of everything, sitting in the same table, working exactly as designed. None of my nine serving queries filtered on the version column.",
      "Confirmed against live data once I knew to look: 1,830 raw match rows for 610 real matches. 15 competition rows for 5 competitions. One match's shots joined out to 147 rows for 49 real shots. Exactly threefold, every time.",
      "The fix is one clause on nine queries, plus ten regression tests so it can't come back quietly.",
      "The same duplication had already bitten the modelling side. An xG join fanned out to 47,211 rows instead of 15,737. I caught that one before anything was written, checked the value was identical across all three copies, and dropped in a distinct.",
      "Here's what actually bothers me about it. The bug was invisible because it was consistent. Nothing errored. No page broke. Every figure was plausible, every leaderboard ranked in the right order, every ratio was fine because both sides were inflated equally. If you'd asked me to eyeball the site for data problems I'd have said it was clean.",
      "I found it by accident, reading raw row counts while building something unrelated.",
      "The lesson I'm taking is that plausible is not the same as correct, and consistent wrongness is the hardest kind to spot. Now I count raw rows against distinct rows before I trust an aggregate, even when there's no reason to think anything's off.",
    ],
  },
  {
    slug: "cxg-v1-to-v3",
    category: "Release notes",
    headline: "CxG v1 to v3: what actually changed",
    takeaway:
      "Three versions, real but modest gains, and two places where I had to correct my own earlier conclusions.",
    date: "2026-08-26",
    readingTime: "4 min",
    body: [
      "Before any model, I built a deliberately stupid one: predict the training-set goal rate for every shot, no features at all. 0.1047 on the event-wide track. I reported its AUC as null rather than 0.5, because 0.5 makes a constant look like it's doing something.",
      "v1 was a kitchen sink. The full candidate pool, additive logistic regression, no interactions, no components. Two of its five event-wide features weren't statistically significant and I kept them on purpose, because the point of a baseline is that you don't get to tune it. Same rule on CxG+: the three defensive-index features were all non-significant, at p values of 0.89, 0.19 and 0.21, and all three stayed in.",
      "v3 added three rolling-window defensive features and the event-wide track's first confirmed interaction. Nothing was removed.",
      "The numbers on the test split, 2,427 shots. Dumb baseline: log loss 0.3281, Brier 0.0911. v1: 0.3058, 0.0872, AUC 0.6939. v3: 0.3003, 0.0852, AUC 0.7148. StatsBomb: 0.2597, 0.0718, AUC 0.7972.",
      "So v3 over v1 is log loss down 1.8%, Brier down 2.3%, AUC up 0.021. The gap to StatsBomb closed from 17.72% to 15.62%. Real and consistent, and genuinely modest. I'm not calling it more than that.",
      "CxG+ went v1 to v2 to v3 and the shape is clear diminishing returns. v2 to v3 moved log loss from 0.2566 to 0.2555 on three extra features. The gap to StatsBomb went from 5.60% to 5.15%.",
      "Two things I had to correct about my own earlier work.",
      "First: v2 found plain logistic regression was literally singular on its feature pool and fell back to ridge. When I built v3 I re-tested instead of assuming that still held, and on the larger pool plain logistic fit with no error at all. Ridge still shipped, but on validation performance, not because it was forced.",
      "Second: on the event-wide track, ridge beat plain logistic by 0.00001 on validation log loss. That's a tie. I wrote it down as a tie rather than as evidence for ridge.",
      "There's a third one I'll own. The brief I wrote for myself said v1's gap to StatsBomb was 15%. Re-measuring gave 17.7% on test and 25.0% on validation. I flagged the discrepancy against my own premise instead of quietly making the number match.",
    ],
  },
  {
    slug: "publish-marker-bug",
    category: "Dev log",
    headline: "The publish marker bug, and what I got wrong writing it up",
    takeaway:
      "A publication-ordering defect held up acceptance for two days. Going back through git for this writeup, my own incident note didn't hold up either.",
    date: "2026-08-26",
    readingTime: "4 min",
    body: [
      "When the Silver layer publishes, it writes parquet files, a manifest, and a marker file called _SUCCESS to cloud storage. The marker is a promise: this publish finished, it's safe to read.",
      "The defect was that the ordering of that upload wasn't actually guaranteed. In principle a reader could see the completion marker sitting on top of an incomplete publish and treat it as done.",
      "Nothing downstream had visibly broken. That wasn't the point. Acceptance sat conditional from 19 August, and everything built on top of that output, the Gold layer, the CxG analysis, the defensive-index work, the profile clustering, was resting on a foundation I couldn't certify.",
      "I fixed it in four steps. Patch the uploader to enforce ordering. Republish clean rather than patching in place. Verify the warehouse reconciled against the new output. Run the full suite.",
      "The patch makes the order explicit: parquet first, manifest next, marker last. Previously the data and the manifest sat in one bucket that happened to sort alphabetically, which is not the same thing as being ordered.",
      "I kept the old prefix rather than overwriting it, and published the new one alongside. 72 objects each. If you're going to claim a publish was defective you should keep the defective publish.",
      "Reconciliation came back clean: all 18 governed tables identical, 2,156,823 events, 15,737 shots.",
      "Now the part I'd rather not include.",
      "Going back through git to write this up, the pre-fix code already uploaded the marker last. It had done since the file was written. What my patch actually added was a guarantee that parquet lands before the manifest, which is a real ordering defect but not the one my incident note describes.",
      "So either the bad publish ran from code I hadn't committed, or I wrote the note loosely at the time and it hardened into fact through repetition. I can't tell which from here, and I'm not going to guess.",
      "I've rewritten this as a publication-ordering defect, which is what I can actually prove from the diff, rather than repeating a claim about the marker being written first that the history doesn't support.",
      "That's the more useful story anyway. The original bug cost me two days. The incident note being subtly wrong for a week, and me only catching it because I sat down to write it up properly, is the thing worth remembering.",
      "One postscript. The immutability guard I added during this fix, the one that refuses to republish over a completed prefix, later broke the orchestration chain: every scheduled run failed at the Silver step whenever ingest found nothing new. That was the guard doing exactly its job. I moved the check up into the runner so it no-ops there, and left the builder's hard failure intact for anyone calling it directly.",
    ],
  },
];
