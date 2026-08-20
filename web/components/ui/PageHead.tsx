export function PageHead({ title, crumb }: { title: string; crumb: string }) {
  return (
    <div className="flex items-baseline justify-between mb-[18px] flex-wrap gap-2.5">
      <div>
        <h1 className="text-lg font-semibold m-0">{title}</h1>
        <div className="text-[12.5px] text-muted">{crumb}</div>
      </div>
    </div>
  );
}
