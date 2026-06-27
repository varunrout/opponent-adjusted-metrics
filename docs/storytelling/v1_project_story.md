# V1 Project Story

## The Narrative

Football actions are not equal just because they share the same event type. A shot from a central pocket is different from a hopeful effort from distance. A pass that breaks a line is different from a sideways pass under no pressure. A carry into the box is different from a carry away from goal.

This project turns raw event data into contextual football metrics that are easier to review, compare, and explain. The goal is not to claim a finished production analytics platform. The goal is to show a reproducible modelling foundation that turns football events into interpretable player, team, and action-level insight.

## Why Context Matters

This is why contextual and opponent-adjusted metrics matter: they keep the football situation attached to the event instead of treating every shot, pass, carry, or chance as interchangeable.

Traditional event counts can be useful, but they flatten the game. Shot totals, assists, completed passes, and carries all miss context. A football analyst usually wants to know:

- Was the shot actually dangerous?
- Did the action create a chance or just keep possession?
- Did the movement advance the ball into a more threatening state?
- Did a player add repeatable value, or was the output driven by one unusual event?

Opponent-adjusted and contextual metrics matter because they push the analysis closer to the football question. They help move from "what happened?" toward "how valuable was it?"

## What Each Metric Answers

### CxG

CxG is contextual expected goals. It focuses on shots.

It answers:

- How good was this shot?
- Which players and teams generate high-quality shots?
- How do shot context and opponent information affect scoring probability?

CxG is the shot-quality layer. It should be read as a baseline model with validation and API integration, not as a final production calibration system.

### CxA

CxA is contextual expected assist or chance creation. It focuses on actions that create or progress toward shots.

It answers:

- Which actions created chance value?
- Which players help create shots before the final shot event?
- Which teams generate chance value through build-up and attacking actions?

CxA is the chance-creation layer. It gives analysts a way to credit attacking contribution before the shot, while staying honest that the current implementation is a baseline event-data model and attribution layer.

### CxT

CxT is contextual expected threat. In the current baseline, it values moving the ball into more threatening pitch zones.

It answers:

- Which actions moved the ball into more dangerous areas?
- Which players add threat through passing and carrying?
- Which teams progress the ball into valuable zones?
- Which zone transitions create the most threat?

CxT is the territorial progression layer. The current implementation is a deterministic baseline grid model. CxT+, Advanced CxT, and opponent defensive adjusted CxT are future roadmap items.

## How The Metrics Work Together

The three metrics describe different parts of attacking value:

- CxG explains the value of the shot.
- CxA explains the value of actions that create or lead toward chances.
- CxT explains the value of moving the ball into threatening territory.

Together, they let a reviewer tell a fuller football story. A player might rank low in CxG because they do not shoot, but high in CxA because they create chances. Another player might rank high in CxT because they consistently progress the ball into dangerous zones, even if they do not play the final pass.

At team level, the combination can reveal style. A team with high CxT but lower CxG may progress well but fail to convert territory into strong shots. A team with high CxG but lower CxT may rely on fewer, sharper attacks. These are the kinds of questions the v1 dashboard should make easy to inspect.

## From Raw Events To Insight

The project flow is:

1. Ingest or load football event data.
2. Build metric-specific feature tables.
3. Generate predictions, aggregates, reports, and metadata under ignored output directories.
4. Validate model and output contracts.
5. Use dashboard-ready tables and reports to explain player, team, and action value.

The generated files are intentionally not committed. The repository tracks the code, contracts, docs, and tests that make the outputs reproducible.

## How An Analyst Would Use It

A football analyst could use the project to:

- Start with a team view and see whether value comes from shots, chance creation, or progression.
- Open a player profile and compare CxG, CxA, and baseline CxT contribution.
- Drill into action-level rows to inspect high-value passes, carries, and box entries.
- Use CxT zone-transition reports to understand where threat is created.
- Use validation and diagnostics pages to check what is implemented, what is generated locally, and what remains a roadmap item.

The strongest v1 experience is not a black-box model demo. It is a clear, traceable analytics story: raw events become reproducible metrics, metrics become aggregate reports, and reports become football insight.

## What V1 Should Feel Like

V1 should feel like a compact football analysis product. It should be fast to understand, honest about limitations, and useful for asking real football questions. It should not hide behind modelling jargon, and it should not overclaim that baseline metrics are production-grade.

The reviewer should leave with three takeaways:

- The modelling outputs are reproducible.
- The metrics answer distinct football questions.
- The project is ready for dashboard implementation and deeper post-v1 modelling work.

## Dashboard Demo Story

The v1 dashboard should be demoed in this order:

1. Overview: show the product goal, metric explanations, output availability, and v1 status.
2. Player analysis: explain what a strong player looks like across CxG, CxA, and CxT.
3. Team analysis: show how teams create value through shots, chance creation, or progression.
4. CxG: explain shot quality.
5. CxA: explain chance-creation action value.
6. CxT: explain threat added by ball progression.
7. Action explorer: trace aggregate value back to individual events.
8. Reports / diagnostics: show generated-output status and limitations.

The demo should explicitly say that CxT+, Contextual CxT, Advanced CxT, and OD-CxT are deferred until after v1. Future screenshots or GIFs should capture the overview banner, player/team tables, CxT interpretation reports, action explorer, and missing-output state.
