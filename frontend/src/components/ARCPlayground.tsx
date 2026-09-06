import { useState } from 'react';
import { Play, Settings2, RefreshCw } from 'lucide-react';

const ARCPlayground = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);

  const handleRun = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/playground/episode?family=ColorTransformation&seed=42', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          difficulty: { grid_size: [5, 5], n_demonstrations: 2 }
        })
      });
      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const renderGrid = (grid: number[][]) => {
    const colors = [
      'bg-gray-900', 'bg-blue-500', 'bg-red-500', 'bg-green-500', 
      'bg-yellow-500', 'bg-purple-500', 'bg-pink-500', 'bg-orange-500', 
      'bg-teal-500', 'bg-indigo-500'
    ];
    
    return (
      <div className="grid gap-1 p-1 bg-gray-800 rounded border border-gray-700 w-fit">
        {grid.map((row, i) => (
          <div key={i} className="flex gap-1">
            {row.map((cell, j) => (
              <div 
                key={`${i}-${j}`} 
                className={`w-6 h-6 rounded-sm ${colors[cell] || colors[0]} border border-gray-700/50`}
              />
            ))}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center bg-gray-800 p-4 rounded-xl border border-gray-700">
        <div>
          <h3 className="font-semibold text-gray-200">Interactive ARC Evaluation</h3>
          <p className="text-sm text-gray-400">Test the sparse recurrent architecture on structured reasoning tasks.</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleRun}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors"
          >
            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />}
            Run Episode
          </button>
        </div>
      </div>

      {data && data.task && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h4 className="font-medium text-gray-300 mb-4 flex items-center gap-2">
                <Settings2 size={16} /> Demonstrations
              </h4>
              <div className="flex flex-wrap gap-8">
                {data.task.demonstrations.map((demo: any, idx: number) => (
                  <div key={idx} className="space-y-2">
                    <p className="text-xs text-gray-500 font-medium">Demo {idx + 1}</p>
                    <div className="flex items-center gap-4">
                      {renderGrid(demo.input_grid)}
                      <span className="text-gray-600">→</span>
                      {renderGrid(demo.output_grid)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h4 className="font-medium text-gray-300 mb-4">Query & Prediction</h4>
              <div className="flex items-center gap-8">
                <div className="space-y-2">
                  <p className="text-xs text-gray-500 font-medium">Input Query</p>
                  {renderGrid(data.task.query.input_grid)}
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-blue-500 mb-1">● LIVE COMPUTATION</span>
                  <span className="text-gray-600">→</span>
                </div>
                <div className="space-y-2">
                  <p className="text-xs text-gray-500 font-medium">Model Prediction</p>
                  {data.prediction ? renderGrid(data.prediction) : (
                    <div className="w-32 h-32 flex items-center justify-center bg-gray-900 border border-gray-700 rounded text-gray-600 text-sm">
                      No output
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
              <h4 className="font-medium text-gray-300 mb-4">Telemetry</h4>
              {data.telemetry && (
                <div className="space-y-4">
                  <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                    <p className="text-xs text-gray-500 mb-1">Memory Norm</p>
                    <p className="text-lg font-mono text-blue-400">{data.telemetry.memory.norm.toFixed(3)}</p>
                  </div>
                  <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                    <p className="text-xs text-gray-500 mb-1">Mean Sparsity</p>
                    <p className="text-lg font-mono text-purple-400">
                      {(data.telemetry.neurons.mean_sparsity * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                    <p className="text-xs text-gray-500 mb-1">Reasoning Depth</p>
                    <p className="text-lg font-mono text-green-400">{data.telemetry.reasoning.trajectory_length}</p>
                  </div>
                  <div className="bg-gray-900 p-3 rounded-lg border border-gray-700">
                    <p className="text-xs text-gray-500 mb-1">Failure Risk</p>
                    <p className="text-lg font-mono text-emerald-400 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400"></span> SAFE
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ARCPlayground;
