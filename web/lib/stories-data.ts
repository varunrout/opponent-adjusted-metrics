// Embedded analysis figures — charts/diagrams/tables placed inline at the
// point in the prose where the numbers are actually discussed, per the
// "blog with embedded figures" story-page format. Kept as typed data here
// (not JSX) for the same reason `body` is plain strings: this stays a data
// file, and the story page maps `kind` to a presentational component in
// `components/story/`. Every number a figure shows must already appear in
// that story's `body` prose — figures illustrate, they don't introduce.
export type StoryFigure =
  | {
      kind: "split-calc";
      afterParagraph: number;
      title: string;
      caption?: string;
      shared: { label: string; value: string }[];
      branches: { label: string; result: string }[];
    }
  | {
      kind: "grouped-bars";
      afterParagraph: number;
      title: string;
      caption?: string;
      rows: {
        label: string;
        bars: { label: string; value: number; color: string }[];
      }[];
    }
  | {
      kind: "copies-diagram";
      afterParagraph: number;
      title: string;
      caption?: string;
      copyLabels: string[];
      perCopyValue: number;
      entityLabel: string;
      totalLabel: string;
      note: string;
    }
  | {
      kind: "table";
      afterParagraph: number;
      title: string;
      caption?: string;
      columns: string[];
      rows: string[][];
    }
  | {
      kind: "pass-fail-list";
      afterParagraph: number;
      title: string;
      caption?: string;
      items: { label: string; passed: boolean }[];
    }
  | {
      kind: "upload-timeline";
      afterParagraph: number;
      title: string;
      caption?: string;
      claimed: { label: string }[];
      actual: { label: string; detail?: string }[];
    };

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
  // Rendered by the story page after `body[figure.afterParagraph]`.
  figures?: StoryFigure[];
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
    figures: [
      {
        kind: "pass-fail-list",
        afterParagraph: 3,
        title: "Univariate screen: 7 candidate features",
        caption:
          "Six of seven cleared the sign-stability and minimum-correlation thresholds on the event-wide track. Match minute was the one that didn't — later further trimmed to five for redundancy.",
        items: [
          { label: "Score difference", passed: true },
          { label: "Game state", passed: true },
          { label: "Match minute", passed: false },
          { label: "Regulation time remaining", passed: true },
          { label: "Manpower difference", passed: true },
          { label: "Late-game trailing", passed: true },
          { label: "Late-game leading", passed: true },
        ],
      },
      {
        kind: "table",
        afterParagraph: 5,
        title: "Bivariate test: 3 defensive-index × 5 match-context features, 15 pairs",
        caption: "None of the fifteen pairs validated on the held-out split — including the three that looked significant on training data alone.",
        columns: ["Pair", "Training p-value", "Held-out"],
        rows: [
          ["GK index × manpower difference", "0.0145", "Failed"],
          ["Nearest-defender index × manpower difference", "0.0292", "Failed"],
          ["Nearest-defender index × late-game leading", "0.0327", "Failed"],
          ["Remaining 12 pairs", "Not significant on training", "Failed"],
        ],
      },
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
    figures: [
      {
        kind: "copies-diagram",
        afterParagraph: 1,
        title: "Root cause: 3 schema-version copies, 0 of 9 queries filtered",
        caption:
          "Every core table carries one full copy of every row per schema version. None of the nine serving queries filtered on it, so every aggregate summed all three.",
        copyLabels: ["statsbomb_silver_v1", "statsbomb_silver_v1_1", "statsbomb_silver_v1_2"],
        perCopyValue: 610,
        entityLabel: "matches",
        totalLabel: "rows returned by an unfiltered query",
        note: "0 of 9 serving queries filtered on schema version",
      },
      {
        kind: "grouped-bars",
        afterParagraph: 2,
        title: "Real vs. shown on site",
        caption:
          "Exactly threefold, every time — the signature of an unfiltered join over 3 identical schema-version copies.",
        rows: [
          {
            label: "Matches",
            bars: [
              { label: "Real", value: 610, color: "var(--teal)" },
              { label: "Shown", value: 1830, color: "var(--red)" },
            ],
          },
          {
            label: "Competitions",
            bars: [
              { label: "Real", value: 5, color: "var(--teal)" },
              { label: "Shown", value: 15, color: "var(--red)" },
            ],
          },
          {
            label: "One match's shots",
            bars: [
              { label: "Real", value: 49, color: "var(--teal)" },
              { label: "Shown", value: 147, color: "var(--red)" },
            ],
          },
        ],
      },
    ],
  },
  {
    slug: "same-gap-two-percentages",
    category: "Methodology",
    headline: "The gap to StatsBomb was 15%. It was also 17.7%. Both were right.",
    takeaway:
      "Two of my own reports quoted a different gap to StatsBomb xG for the same model. Neither was wrong. I'd just never written down which number was on the bottom.",
    date: "2026-08-26",
    readingTime: "3 min",
    body: [
      "I went back to check a number and found two versions of it. The original v1 report said CxG event-wide's gap to StatsBomb xG was 15%. A later report, written by me, working from the same model, said 17.7%.",
      "Same model. Same test split. Same underlying log-loss values. Different answer.",
      "The first report computed (v1 log-loss minus StatsBomb log-loss) divided by v1's own log-loss: (0.3058 - 0.2597) / 0.3058, which is 15.08%. The second computed the same subtraction divided by StatsBomb's log-loss instead: (0.3058 - 0.2597) / 0.2597, which is 17.72%. Both are correct arithmetic. They just answer different questions. One asks how much lower StatsBomb is, relative to my model. The other asks how much higher my model is, relative to StatsBomb.",
      "I didn't just assume the second convention was fine because it felt right. I checked it against a number I already trusted: CxG+'s published gap of 5.6%. Running the same StatsBomb-denominator formula on CxG+'s v2 numbers reproduces 5.60% exactly. That's the convention this site has actually been using, whether I'd written it down or not.",
      "There was a third number sitting next to these two that looked like it belonged to the same problem and didn't. A validation-split figure of 25.0% for the same model. I went looking for what it contradicted and found nothing, because the original report never published a validation-split comparison for this track at all. It wasn't a conflicting number. It was a number that hadn't existed before.",
      "I checked whether the underlying data had quietly changed between the two reports, because that's the boring explanation and boring explanations are usually right. It hadn't. The table backing both reports carries one materialized_at timestamp, never regenerated. Bit-for-bit the same values, read twice, described two different ways.",
      "So the fix isn't a number, it's a habit. I'm not allowed to write a bare gap percentage anymore. Every one from here gets the split it was measured on and which value sat on the bottom of the fraction. '17.7% higher log-loss than StatsBomb, relative to StatsBomb, test split' is clumsier to write than '17.7% gap.' It's also the only version of that sentence that means something on its own a year from now.",
    ],
    figures: [
      {
        kind: "split-calc",
        afterParagraph: 2,
        title: "Same two numbers, two denominators, two correct answers",
        caption:
          "The subtraction (0.3058 − 0.2597) is identical both times. Dividing by a different one of the two source numbers changes only what the percentage is relative to — not which arithmetic is right.",
        shared: [
          { label: "v1 log-loss", value: "0.3058" },
          { label: "StatsBomb log-loss", value: "0.2597" },
        ],
        branches: [
          { label: "(0.3058 − 0.2597) ÷ v1's own log-loss (0.3058)", result: "15.08%" },
          { label: "(0.3058 − 0.2597) ÷ StatsBomb log-loss (0.2597)", result: "17.72%" },
        ],
      },
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
    figures: [
      {
        kind: "table",
        afterParagraph: 6,
        title: "Closure gates",
        columns: ["Gate", "Result"],
        rows: [
          [
            "Ordered-upload fix verified (code trace + empirical timestamps)",
            "PASS — 0 objects created after _SUCCESS",
          ],
          ["Full test suite", "PASS — 198/198"],
          [
            "Fresh compliant publish, old prefix preserved",
            "PASS — 72 objects each, old _SUCCESS still dated 2026-08-19 18:23:25 UTC",
          ],
          ["oam_core reconciliation", "PASS — 18/18 tables, 2,156,823 events, 15,737 shots"],
        ],
      },
      {
        kind: "upload-timeline",
        afterParagraph: 8,
        title: "What the incident note claimed vs. what the diff shows",
        caption:
          "The incident note described one defect. The diff shows a different one — real, but not the one that got written down.",
        claimed: [
          { label: "_SUCCESS marker written first" },
          { label: "Everything after it undefined / unordered" },
        ],
        actual: [
          { label: "parquet_files uploaded" },
          { label: "manifest.json uploaded", detail: "12:01:50.152 UTC" },
          { label: "_SUCCESS uploaded last", detail: "12:01:50.215 UTC" },
        ],
      },
    ],
  },
];
