import React from "react";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "neutral" | "success" | "warning" | "error" | "blue";
  size?: "sm" | "md";
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "neutral",
  size = "sm",
  className = "",
}) => {
  const sizeClasses =
    size === "sm"
      ? "text-[11px] px-2 py-0.5"
      : "text-xs px-2.5 py-1 font-medium";

  const variantClasses = {
    neutral: "bg-slate-100 text-slate-700 border border-slate-200/80",
    default: "bg-slate-100 text-slate-800 border border-slate-200",
    success: "bg-emerald-50 text-emerald-700 border border-emerald-200/70",
    warning: "bg-amber-50 text-amber-700 border border-amber-200/70",
    error: "bg-rose-50 text-rose-700 border border-rose-200/70",
    blue: "bg-blue-50 text-blue-700 border border-blue-200/70",
  }[variant];

  return (
    <span
      className={`inline-flex items-center gap-1 font-mono rounded-md font-medium leading-none ${sizeClasses} ${variantClasses} ${className}`}
    >
      {children}
    </span>
  );
};

export const FileTypeBadge: React.FC<{ fileType?: string | null }> = ({ fileType }) => {
  const type = (fileType || "TXT").toUpperCase().replace(/^\./, "");
  let variant: "neutral" | "error" | "blue" | "warning" = "neutral";

  if (type === "PDF") variant = "error";
  else if (type.includes("DOC") || type.includes("DOCX") || type.includes("WORD")) variant = "blue";
  else if (type === "TXT" || type === "MD" || type === "MARKDOWN") variant = "neutral";

  const displayType = type.includes("WORD") || type.includes("OFFICEDOCUMENT") ? "DOCX" : type.slice(0, 5);

  return <Badge variant={variant}>{displayType}</Badge>;
};

export const SimilarityBadge: React.FC<{ score: number }> = ({ score }) => {
  const pct = Math.round(score * 100);
  let variant: "success" | "blue" | "warning" = "warning";
  let label = "Fair";

  if (score >= 0.60) {
    variant = "success";
    label = "High";
  } else if (score >= 0.35) {
    variant = "blue";
    label = "Good";
  }

  return (
    <Badge variant={variant} size="md">
      <span>{pct}%</span>
      <span className="text-[10px] opacity-75 font-sans font-normal ml-0.5">({label})</span>
    </Badge>
  );
};
