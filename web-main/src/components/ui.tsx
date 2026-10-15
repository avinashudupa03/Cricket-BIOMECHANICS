import React from 'react';
import { Loader2 } from 'lucide-react';

export function Card({
  children,
  className = '',
  title,
  subtitle,
  action,
}: {
  children: React.ReactNode;
  className?: string;
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div
      className={`rounded-2xl border border-line bg-surface shadow-[0_1px_3px_rgba(15,23,42,0.04)] ${className}`}
    >
      {(title || action) && (
        <div className="flex items-start justify-between gap-4 px-6 pt-5">
          <div>
            {title && (
              <h3 className="font-display text-[15px] font-semibold text-ink">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="mt-0.5 text-[13px] text-ink-3">{subtitle}</p>
            )}
          </div>
          {action}
        </div>
      )}
      <div className={title || action ? 'p-6 pt-4' : 'p-6'}>{children}</div>
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  tone?: 'neutral' | 'brand' | 'amber' | 'red' | 'violet' | 'blue';
}) {
  const tones: Record<string, string> = {
    neutral: 'bg-surface-2 text-ink',
    brand: 'bg-brand-50 text-brand-600',
    amber: 'bg-amber-50 text-amber-600',
    red: 'bg-red-50 text-red-600',
    violet: 'bg-violet-50 text-violet-600',
    blue: 'bg-blue-50 text-blue-600',
  };
  return (
    <div className="rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_3px_rgba(15,23,42,0.04)]">
      <p className="text-[12px] font-medium uppercase tracking-wide text-ink-3">
        {label}
      </p>
      <p
        className={`mt-2 font-display text-[26px] font-bold leading-tight ${tones[tone]?.split(' ')[1] ?? 'text-ink'} inline-block rounded-lg px-2 py-0.5 ${tones[tone]?.split(' ')[0] ?? ''}`}
      >
        {value}
      </p>
      {hint && <p className="mt-1.5 text-[13px] text-ink-3">{hint}</p>}
    </div>
  );
}

export function Badge({
  children,
  tone = 'slate',
}: {
  children: React.ReactNode;
  tone?: 'slate' | 'brand' | 'amber' | 'red' | 'violet' | 'blue';
}) {
  const tones: Record<string, string> = {
    slate: 'bg-slate-100 text-slate-700',
    brand: 'bg-emerald-100 text-emerald-700',
    amber: 'bg-amber-100 text-amber-700',
    red: 'bg-red-100 text-red-700',
    violet: 'bg-violet-100 text-violet-700',
    blue: 'bg-blue-100 text-blue-700',
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11.5px] font-semibold ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  variant = 'primary',
  className = '',
  type = 'button',
  disabled,
  onClick,
}: {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  className?: string;
  type?: 'button' | 'submit';
  disabled?: boolean;
  onClick?: () => void;
}) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-[13.5px] font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50';
  const variants: Record<string, string> = {
    primary:
      'bg-brand text-white hover:bg-brand-600 shadow-sm hover:shadow focus:ring-emerald-200',
    secondary:
      'bg-surface border border-line text-ink-2 hover:bg-surface-2 hover:text-ink focus:ring-slate-200',
    ghost:
      'bg-transparent text-ink-2 hover:bg-surface-3 focus:ring-slate-200',
    danger: 'bg-red-600 text-white hover:bg-red-700 shadow-sm focus:ring-red-200',
  };
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`${base} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Spinner({ className = 'h-4 w-4' }: { className?: string }) {
  return <Loader2 className={`animate-spin ${className}`} />;
}

export function ProgressBar({
  value,
  color = 'bg-brand',
}: {
  value: number;
  color?: string;
}) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
      <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${v}%` }} />
    </div>
  );
}

export function Ring({ value, size = 120 }: { value: number; size?: number }) {
  const stroke = 10;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(10, value));
  const pct = v / 10;
  const color = v >= 7.5 ? '#059669' : v >= 5 ? '#d97706' : '#dc2626';
  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-[26px] font-bold text-ink">
          {v.toFixed(1)}
        </span>
        <span className="text-[11px] text-ink-3">/ 10</span>
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  message,
  children,
}: {
  icon?: React.ReactNode;
  title: string;
  message?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-surface py-14 text-center">
      {icon && (
        <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-50 text-brand">
          {icon}
        </div>
      )}
      <h3 className="font-display text-[16px] font-semibold text-ink">{title}</h3>
      {message && (
        <p className="mt-1 max-w-sm text-[13.5px] text-ink-3">{message}</p>
      )}
      {children && <div className="mt-4">{children}</div>}
    </div>
  );
}

export function MetricRow({
  label,
  value,
  unit,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-xl bg-surface-2 px-4 py-3">
      <span className="text-[13px] text-ink-2">{label}</span>
      <span className="font-display text-[15px] font-semibold text-ink">
        {value}
        {unit && <span className="ml-1 text-[12px] font-normal text-ink-3">{unit}</span>}
      </span>
    </div>
  );
}

export function SectionHeading({
  title,
  subtitle,
  action,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 className="font-display text-[20px] font-bold text-ink">{title}</h2>
        {subtitle && <p className="mt-0.5 text-[13.5px] text-ink-3">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function valueOrDash(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}