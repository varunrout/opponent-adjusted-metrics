import type { StoryFigure } from "@/lib/stories-data";
import { StoryFigureCard } from "@/components/story/StoryFigureCard";
import { SplitCalcDiagram } from "@/components/story/SplitCalcDiagram";
import { GroupedBars } from "@/components/story/GroupedBars";
import { CopiesDiagram } from "@/components/story/CopiesDiagram";
import { SimpleTable } from "@/components/ui/SimpleTable";
import { PassFailList } from "@/components/story/PassFailList";
import { UploadTimeline } from "@/components/story/UploadTimeline";

// Maps a data-only StoryFigure spec (lib/stories-data.ts) to its
// presentational component, so the story page and the data file stay
// decoupled from the figure-kind-to-component mapping.
export function StoryFigureView({ figure }: { figure: StoryFigure }) {
  return (
    <StoryFigureCard title={figure.title} caption={figure.caption}>
      {figure.kind === "split-calc" && (
        <SplitCalcDiagram shared={figure.shared} branches={figure.branches} />
      )}
      {figure.kind === "grouped-bars" && <GroupedBars rows={figure.rows} />}
      {figure.kind === "copies-diagram" && (
        <CopiesDiagram
          copyLabels={figure.copyLabels}
          perCopyValue={figure.perCopyValue}
          entityLabel={figure.entityLabel}
          totalLabel={figure.totalLabel}
          note={figure.note}
        />
      )}
      {figure.kind === "table" && <SimpleTable columns={figure.columns} rows={figure.rows} />}
      {figure.kind === "pass-fail-list" && <PassFailList items={figure.items} />}
      {figure.kind === "upload-timeline" && (
        <UploadTimeline claimed={figure.claimed} actual={figure.actual} />
      )}
    </StoryFigureCard>
  );
}
