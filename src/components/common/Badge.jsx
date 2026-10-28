import "./Badge.css";

/**
 * Reusable badge for provenance (REAL, DERIVED, ILLUSTRATIVE, SIMULATED, UNAVAILABLE)
 * or general stage statuses (active, complete, idle, error).
 */
export default function Badge({
  type = "provenance", // 'provenance' | 'status' | 'neutral'
  variant = "illustrative", // 'real' | 'derived' | 'illustrative' | 'simulated' | 'unavailable' | 'active' | 'complete' | 'idle' | 'error'
  label,
  size = "md", // 'sm' | 'md'
}) {
  const displayLabel = label || variant.toUpperCase();

  return (
    <span
      className={`badge badge--${type} badge--${variant.toLowerCase()} badge--${size}`}
      title={`Data provenance: ${displayLabel}`}
    >
      <span className="badge__dot" />
      <span className="badge__text">{displayLabel}</span>
    </span>
  );
}
