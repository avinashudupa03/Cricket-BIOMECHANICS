import React from 'react';
import {
  LayoutGrid,
  UploadCloud,
  History,
  BrainCircuit,
  FileBarChart2,
  Activity,
  CircleDot,
} from 'lucide-react';
import { NavRoute } from '../types';

const NAV_ITEMS: {
  route: NavRoute;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { route: 'dashboard', label: 'Dashboard', icon: LayoutGrid },
  { route: 'analyze', label: 'Upload & Analyze', icon: UploadCloud },
  { route: 'history', label: 'Analysis History', icon: History },
  { route: 'ml', label: 'ML Insights', icon: BrainCircuit },
  { route: 'reports', label: 'Reports & EDA', icon: FileBarChart2 },
];

export function Sidebar({
  currentRoute,
  onRouteChange,
}: {
  currentRoute: NavRoute;
  onRouteChange: (r: NavRoute) => void;
}) {
  return (
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 flex-col bg-nav text-slate-300 lg:flex">
      <div className="flex items-center gap-3 px-6 py-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand text-white shadow-lg shadow-brand/30">
          <Activity className="h-5 w-5" />
        </div>
        <div className="leading-tight">
          <p className="font-display text-[15px] font-bold text-white">
            Cricket Biomechanics
          </p>
          <p className="text-[11px] tracking-wide text-slate-400">AI Batting Analysis</p>
        </div>
      </div>

      <div className="px-4 pt-2 pb-4 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        Platform
      </div>

      <nav className="flex-1 space-y-1 px-4">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = currentRoute === item.route;
          return (
            <button
              key={item.route}
              onClick={() => onRouteChange(item.route)}
              className={`group flex w-full items-center gap-3 rounded-xl px-3.5 py-2.5 text-[13.5px] font-medium transition-all ${
                active
                  ? 'bg-brand text-white shadow-md shadow-brand/25'
                  : 'text-slate-300 hover:bg-nav-2 hover:text-white'
              }`}
            >
              <Icon className={`h-[18px] w-[18px] ${active ? '' : 'opacity-70 group-hover:opacity-100'}`} />
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="mx-4 mb-4 rounded-xl border border-white/10 bg-nav-2 p-4">
        <div className="flex items-center gap-2 text-emerald-400">
          <CircleDot className="h-3.5 w-3.5 animate-pulse" />
          <span className="text-[12px] font-semibold">Pipeline ready</span>
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed text-slate-400">
          MediaPipe pose estimation runs locally. Upload a batting clip to begin.
        </p>
      </div>
    </aside>
  );
}