import React from 'react';
import { Menu } from 'lucide-react';
import { NavRoute } from '../types';

const ROUTE_TITLES: Record<NavRoute, { title: string; sub: string }> = {
  dashboard: { title: 'Dashboard', sub: 'Dataset overview & recent analysis' },
  analyze: { title: 'Upload & Analyze', sub: 'Run pose estimation on a batting clip' },
  results: { title: 'Analysis Results', sub: 'Biomechanics breakdown for one clip' },
  history: { title: 'Analysis History', sub: 'Every processed video, newest first' },
  ml: { title: 'ML Insights', sub: 'Shot classification model evaluation' },
  reports: { title: 'Reports & EDA', sub: 'Exploratory analysis & validation reports' },
};

export function TopBar({
  route,
  onMenuClick,
}: {
  route: NavRoute;
  onMenuClick: () => void;
}) {
  const meta = ROUTE_TITLES[route] ?? ROUTE_TITLES.dashboard;
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-surface/85 backdrop-blur">
      <div className="flex items-center gap-3 px-6 py-4">
        <button
          onClick={onMenuClick}
          className="rounded-lg p-2 text-ink-2 hover:bg-surface-3 lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div>
          <h1 className="font-display text-[19px] font-bold text-ink">{meta.title}</h1>
          <p className="text-[12.5px] text-ink-3">{meta.sub}</p>
        </div>
        <div className="ml-auto text-right text-[11px] leading-tight text-ink-3">
          <p className="font-semibold text-ink-2">Pose: MediaPipe</p>
          <p>Model: LOOCV · seed 42</p>
        </div>
      </div>
    </header>
  );
}