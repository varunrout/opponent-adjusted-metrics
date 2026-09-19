import { PageHead } from "@/components/ui/PageHead";
import { Card } from "@/components/ui/Card";

const GLOSSARY: { term: string; definition: string }[] = [
  {
    term: "xG (expected goals)",
    definition:
      "StatsBomb's own probability, per shot, that it results in a goal — the industry-standard baseline this project compares against.",
  },
  {
    term: "CxG (context-adjusted expected goals)",
    definition:
      "This project's own shot-quality model, trained on the event-wide feature set (8 features: track name cxg_event). Adjusts a shot's value for the defensive context around it — how it responds is disclosed, honestly, in the Models and Stories tabs.",
  },
  {
    term: "CxG+",
    definition:
      "A second CxG track (24 features) that additionally uses StatsBomb's 360 freeze-frame tracking data where available — nearest defender position, backline shape, goalkeeper positioning at the moment of the shot.",
  },
  {
    term: "xG/shot",
    definition: "Total xG divided by shot count for a player or team — average shot quality, not volume.",
  },
  {
    term: "G−xG (goals minus xG)",
    definition:
      "Actual goals scored minus total xG. Positive means finishing above what shot quality alone would predict; negative means below. The headline \"who's overperforming\" number.",
  },
  {
    term: "Opponent-adjusted",
    definition:
      "The project's namesake idea: a shot's quality depends not just on where it was taken from, but on who was defending it — nearest defender distance and role, backline compactness, goalkeeper position. CxG is the model that tries to capture this; a richer per-shot view of the underlying defensive context exists in BigQuery but isn't yet surfaced anywhere in the app (see Limitations).",
  },
];

export default function AboutPage() {
  return (
    <section>
      <PageHead title="About" crumb="What this is, how it's built, and what it doesn't do yet" />

      <div className="flex flex-col gap-4">
        <Card title="What this is">
          <p className="text-[13px] text-text2 m-0 leading-relaxed">
            Opponent-Adjusted Metrics is a from-scratch expected-goals pipeline and dashboard, built end to
            end by one person: data ingestion, feature engineering, model training, a FastAPI backend, and
            this Next.js frontend. Its goal is to test whether a shot&apos;s expected-goals value can be
            improved by adjusting for the defensive context around it, and to publish that test honestly —
            including where it falls short of the industry baseline.
          </p>
        </Card>

        <Card title="Data source">
          <p className="text-[13px] text-text2 m-0 leading-relaxed mb-2">
            All data comes from{" "}
            <a
              href="https://github.com/statsbomb/open-data"
              target="_blank"
              rel="noreferrer"
              className="text-teal hover:underline"
            >
              StatsBomb Open Data
            </a>
            , published under StatsBomb&apos;s own open-data licence for public, non-commercial use. Scope:
            3 competitions (Premier League, FIFA World Cup, UEFA Euro), 5 seasons, 610 matches.
          </p>
          <p className="text-[13px] text-text2 m-0 leading-relaxed">
            StatsBomb xG is StatsBomb&apos;s own published expected-goals model — a well-established,
            independently maintained baseline that this project&apos;s CxG models are compared against
            throughout the app, never replaced by them.
          </p>
        </Card>

        <Card title="Glossary">
          <dl className="grid gap-x-6 gap-y-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
            {GLOSSARY.map((entry) => (
              <div key={entry.term}>
                <dt className="text-[12.5px] font-medium text-text m-0">{entry.term}</dt>
                <dd className="text-[12px] text-muted m-0 mt-1 ml-0 leading-relaxed">{entry.definition}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card title="How it's built">
          <p className="text-[13px] text-text2 m-0 leading-relaxed">
            StatsBomb Open Data flows into a BigQuery medallion architecture: <code>oam_core</code> (silver,
            cleaned event/shot/match data), <code>oam_analysis</code> (feature engineering and exploratory
            statistics), <code>oam_ml</code> (model training artifacts and predictions), and{" "}
            <code>oam_features</code> (a further 9-table feature layer, not yet read by any application
            code). A FastAPI backend on Cloud Run queries these tables directly — filtered, cached, and
            cost-audited — and serves this Next.js frontend, hosted on Firebase Hosting.
          </p>
        </Card>

        <Card title="Honest limitations" className="border-l-[3px]" style={{ borderLeftColor: "var(--amber)" }}>
          <ul className="text-[13px] text-text2 m-0 pl-4 flex flex-col gap-1.5">
            <li>
              CxG is evaluated on a fixed test split, not scored live — there is no real-time inference
              pipeline. Every CxG value shown is a stored prediction, not a fresh model call.
            </li>
            <li>CxG trails the StatsBomb xG baseline on every metric, on both tracks (see Models/Stories).</li>
            <li>CxA (context-adjusted assists) and CxT (context-adjusted threat) are not built yet.</li>
            <li>
              <code>oam_serving</code> — the table a real production serving layer would read from — is
              empty. Nothing here is a production inference system.
            </li>
            <li>Coverage is one season per competition, not multi-season history for any of the three.</li>
          </ul>
        </Card>
      </div>
    </section>
  );
}
