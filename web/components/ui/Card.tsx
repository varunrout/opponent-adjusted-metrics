export function Card({
  children,
  className = "",
  title,
  style,
}: {
  children: React.ReactNode;
  className?: string;
  title?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div className={`bg-card border border-border rounded p-3.5 px-4 ${className}`} style={style}>
      {title && <h3 className="text-[12.5px] font-medium text-text2 m-0 mb-2.5">{title}</h3>}
      {children}
    </div>
  );
}
