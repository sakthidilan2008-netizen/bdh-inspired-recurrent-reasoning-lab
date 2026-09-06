import { Beaker, Info } from 'lucide-react';
import ARCPlayground from './components/ARCPlayground';

function App() {
  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
      <aside className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-6">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            BDH Research Lab
          </h1>

          <p className="text-xs text-gray-400 mt-2">
            DataForge 2026
          </p>
        </div>

        <nav className="flex-1 px-4">
          <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-blue-600/20 text-blue-400 border border-blue-500/30">
            <Beaker size={18} />
            <span className="font-medium">
              Research Playground
            </span>
          </div>

          <div className="mt-6 px-4">
            <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wider">
              <Info size={14} />
              About
            </div>

            <p className="text-[11px] text-gray-600 leading-relaxed mt-3">
              Teaching-scale experimental implementation inspired by
              publicly documented BDH/BDH-CQ principles. This project
              does not claim to reproduce Pathway's proprietary
              production system.
            </p>
          </div>
        </nav>

        <div className="p-4 border-t border-gray-700">
          <p className="text-[10px] text-gray-600 leading-relaxed">
            Interactive research artifact • Controlled ARC-style tasks
          </p>
        </div>
      </aside>

      <main className="flex-1 overflow-auto bg-gray-900">
        <header className="h-16 border-b border-gray-800 flex items-center px-8">
          <h2 className="text-lg font-semibold text-gray-200">
            Research Playground
          </h2>
        </header>

        <div className="p-8 max-w-7xl mx-auto">
          <ARCPlayground />
        </div>
      </main>
    </div>
  );
}

export default App;