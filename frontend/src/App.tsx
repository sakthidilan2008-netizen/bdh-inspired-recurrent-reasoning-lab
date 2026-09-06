import { useState } from 'react';
import { Beaker, BrainCircuit, Database, Navigation, Layers, ShieldAlert, BarChart2 } from 'lucide-react';
import ARCPlayground from './components/ARCPlayground';

function App() {
  const [activeTab, setActiveTab] = useState('arc');

  const tabs = [
    { id: 'arc', label: 'ARC Lab', icon: <Beaker size={18} /> },
    { id: 'memory', label: 'Memory Lab', icon: <Database size={18} /> },
    { id: 'latent', label: 'Latent Reasoning', icon: <Navigation size={18} /> },
    { id: 'neurons', label: 'Neuron Observatory', icon: <BrainCircuit size={18} /> },
    { id: 'failure', label: 'Failure Atlas', icon: <ShieldAlert size={18} /> },
    { id: 'mitigation', label: 'Mitigation Lab', icon: <Layers size={18} /> },
    { id: 'benchmark', label: 'Benchmarks', icon: <BarChart2 size={18} /> },
  ];

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-6">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            BDH Research Lab
          </h1>
          <p className="text-xs text-gray-400 mt-2">DataForge 2026</p>
        </div>
        
        <nav className="flex-1 px-4 space-y-2">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                activeTab === tab.id 
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30' 
                  : 'text-gray-400 hover:bg-gray-700/50 hover:text-gray-200'
              }`}
            >
              {tab.icon}
              <span className="font-medium">{tab.label}</span>
            </button>
          ))}
        </nav>
        
        <div className="p-4 border-t border-gray-700">
          <p className="text-[10px] text-gray-500 leading-tight">
            We implement a transparent experimental reproduction of publicly documented BDH/BDH-CQ principles, systematically evaluate its behavior, and contribute original mechanisms. We do not claim to reproduce Pathway's proprietary production system.
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto bg-gray-900">
        <header className="h-16 border-b border-gray-800 flex items-center px-8">
          <h2 className="text-lg font-semibold text-gray-200">
            {tabs.find(t => t.id === activeTab)?.label}
          </h2>
        </header>
        
        <main className="p-8">
          {activeTab === 'arc' && <ARCPlayground />}
          {activeTab !== 'arc' && (
            <div className="flex flex-col items-center justify-center h-96 text-gray-500 border border-dashed border-gray-700 rounded-xl bg-gray-800/30">
              {tabs.find(t => t.id === activeTab)?.icon}
              <p className="mt-4 text-lg font-medium">Under Construction for Hackathon Submission</p>
              <p className="text-sm text-gray-600 mt-2">Check back after Phase 18 Integration</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
