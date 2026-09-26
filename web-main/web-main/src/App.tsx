import React, { useState } from 'react';
import { X } from 'lucide-react';
import { NavRoute, VideoMeta } from './types';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { DashboardView } from './views/DashboardView';
import { AnalyzeView } from './views/AnalyzeView';
import { ResultsView } from './views/ResultsView';
import { HistoryView } from './views/HistoryView';
import { MLInsightsView } from './views/MLInsightsView';
import { ReportsView } from './views/ReportsView';

export function App() {
  const [route, setRoute] = useState<NavRoute>('dashboard');
  const [activeVideo, setActiveVideo] = useState<VideoMeta | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const navigate = (r: NavRoute) => {
    setRoute(r);
    setDrawerOpen(false);
    window.scrollTo({ top: 0 });
  };

  const openResults = (video: VideoMeta) => {
    setActiveVideo(video);
    navigate('results');
  };

  return (
    <div className="min-h-screen bg-page font-[Inter,sans-serif] text-ink">
      {/* Mobile drawer */}
      {drawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm" onClick={() => setDrawerOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-64">
            <button
              onClick={() => setDrawerOpen(false)}
              className="absolute right-3 top-3 z-10 rounded-lg bg-nav-2 p-2 text-slate-300"
            >
              <X className="h-4 w-4" />
            </button>
            <Sidebar currentRoute={route} onRouteChange={navigate} />
          </div>
        </div>
      )}

      <Sidebar currentRoute={route} onRouteChange={navigate} />

      <div className="lg:pl-64">
        <TopBar route={route} onMenuClick={() => setDrawerOpen(true)} />

        <main className="mx-auto max-w-[1400px] p-6">
          <div key={route} className="fade-in">
            {route === 'dashboard' && (
              <DashboardView onNavigate={navigate} onOpenVideo={openResults} />
            )}
            {route === 'analyze' && <AnalyzeView onOpenVideo={openResults} />}
            {route === 'results' && (
              <ResultsView video={activeVideo} onBack={() => navigate('history')} />
            )}
            {route === 'history' && <HistoryView onOpenVideo={openResults} />}
            {route === 'ml' && <MLInsightsView />}
            {route === 'reports' && <ReportsView />}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;