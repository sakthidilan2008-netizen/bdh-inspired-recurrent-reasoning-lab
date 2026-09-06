import { useState } from 'react';
import { Play, RefreshCw, Settings2, Activity, BrainCircuit, Database } from 'lucide-react';

type Grid = number[][];

interface EpisodeData {
  task: {
    task_id: string;
    family: string;
    seed: number;
    difficulty: {
      grid_size: number[];
      n_demonstrations: number;
    };
    demonstrations: {
      input_grid: Grid;
      output_grid: Grid;
    }[];
    query: {
      input_grid: Grid;
    };
  };
  prediction: Grid;
  evaluation: {
    exact_match: boolean;
    cell_accuracy: number;
    correct_cells: number;
    total_cells: number;
  };
  telemetry: {
    memory: {
      norm: number;
      effective_rank: number;
      entropy: number;
      interference_score: number;
      reconstruction_error: number;
    };
    neurons: {
      mean_sparsity: number;
    };
    reasoning: {
      trajectory_length: number;
    };
  };
  hypothesis_trace?: {
    hypotheses?: unknown[];
  };
  configuration: {
    memory_mode: string;
    reasoning_depth: number;
    adaptive_halt: boolean;
    reread_memory: boolean;
  };
}

const TASKS = [
  'ColorTransformation',
  'Translation',
  'BoundaryPropagation',
];

const MEMORY_MODES = [
  'baseline',
  'simple_hebbian',
  'interference_mitigated',
];

const ARC_COLORS = [
  'bg-gray-950',
  'bg-blue-500',
  'bg-red-500',
  'bg-green-500',
  'bg-yellow-400',
  'bg-purple-500',
  'bg-pink-500',
  'bg-orange-500',
  'bg-teal-500',
  'bg-indigo-500',
];

const ARC_COLOR_NAMES = [
  'Black',
  'Blue',
  'Red',
  'Green',
  'Yellow',
  'Purple',
  'Pink',
  'Orange',
  'Teal',
  'Indigo',
];

const ARCPlayground = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<EpisodeData | null>(null);

  const [family, setFamily] = useState('ColorTransformation');
  const [memoryMode, setMemoryMode] = useState('simple_hebbian');
  const [reasoningDepth, setReasoningDepth] = useState(4);
  const [nDemonstrations, setNDemonstrations] = useState(3);
  const [gridSize, setGridSize] = useState(5);
  const [seed, setSeed] = useState(42);
  const [adaptiveHalt, setAdaptiveHalt] = useState(false);
  const [rereadMemory, setRereadMemory] = useState(true);

  const handleRun = async () => {
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/playground/episode', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          family,
          grid_size: [gridSize, gridSize],
          n_demonstrations: nDemonstrations,
          memory_mode: memoryMode,
          reasoning_depth: reasoningDepth,
          adaptive_halt: adaptiveHalt,
          reread_memory: rereadMemory,
          seed,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error(error);
      alert(
        'Could not connect to the backend. Make sure FastAPI is running on http://localhost:8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 100000));
  };

  const renderGrid = (grid: Grid, compact = false) => {
    const cellSize = compact ? 'w-4 h-4' : 'w-6 h-6';

    return (
      <div className="inline-flex flex-col gap-[2px] p-2 bg-gray-950 rounded-lg border border-gray-700">
        {grid.map((row, i) => (
          <div key={i} className="flex gap-[2px]">
            {row.map((cell, j) => (
              <div
                key={`${i}-${j}`}
                title={`Cell (${i}, ${j}): ${ARC_COLOR_NAMES[cell] ?? cell}`}
                className={`${cellSize} ${
                  ARC_COLORS[cell] || ARC_COLORS[0]
                } rounded-[2px] border border-gray-800`}
              />
            ))}
          </div>
        ))}
      </div>
    );
  };

  const Metric = ({
    label,
    value,
    description,
  }: {
    label: string;
    value: string;
    description?: string;
  }) => (
    <div className="bg-gray-950 border border-gray-700 rounded-lg p-3">
      <p className="text-[11px] uppercase tracking-wide text-gray-500">
        {label}
      </p>
      <p className="text-lg font-mono text-blue-400 mt-1">{value}</p>
      {description && (
        <p className="text-[10px] text-gray-600 mt-1">{description}</p>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Introduction */}
      <section className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-2xl p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-blue-500/10 rounded-xl">
            <BrainCircuit className="text-blue-400" size={28} />
          </div>

          <div>
            <h2 className="text-2xl font-bold text-gray-100">
              Recurrent Reasoning Playground
            </h2>

            <p className="text-gray-400 mt-2 max-w-3xl leading-relaxed">
              Explore a teaching-scale experimental implementation inspired by
              publicly documented BDH and BDH-CQ ideas. Change the task,
              memory, reasoning depth, and seed, then observe the resulting
              computation and diagnostics.
            </p>

            <p className="text-xs text-gray-600 mt-3">
              This is an experimental educational implementation, not
              Pathway's proprietary production system.
            </p>
          </div>
        </div>
      </section>

      {/* Controls */}
      <section className="bg-gray-800 border border-gray-700 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-5">
          <Settings2 size={18} className="text-blue-400" />
          <h3 className="font-semibold text-gray-200">
            Experiment Configuration
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <label className="space-y-2">
            <span className="text-xs text-gray-400">Task Family</span>
            <select
              value={family}
              onChange={(e) => setFamily(e.target.value)}
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
            >
              {TASKS.map((task) => (
                <option key={task} value={task}>
                  {task}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-xs text-gray-400">Memory Mode</span>
            <select
              value={memoryMode}
              onChange={(e) => setMemoryMode(e.target.value)}
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
            >
              {MEMORY_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-xs text-gray-400">
              Reasoning Depth: {reasoningDepth}
            </span>
            <input
              type="range"
              min="1"
              max="16"
              value={reasoningDepth}
              onChange={(e) => setReasoningDepth(Number(e.target.value))}
              className="w-full"
            />
          </label>

          <label className="space-y-2">
            <span className="text-xs text-gray-400">Demonstrations</span>
            <select
              value={nDemonstrations}
              onChange={(e) => setNDemonstrations(Number(e.target.value))}
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
            >
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-xs text-gray-400">Grid Size</span>
            <select
              value={gridSize}
              onChange={(e) => setGridSize(Number(e.target.value))}
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
            >
              {[5, 6, 7, 8, 9, 10].map((n) => (
                <option key={n} value={n}>
                  {n} × {n}
                </option>
              ))}
            </select>
          </label>

          <label className="space-y-2">
            <span className="text-xs text-gray-400">Random Seed</span>
            <div className="flex gap-2">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
              />

              <button
                onClick={randomizeSeed}
                title="Generate random seed"
                className="px-3 bg-gray-700 hover:bg-gray-600 rounded-lg"
              >
                <RefreshCw size={16} />
              </button>
            </div>
          </label>

          <label className="flex items-center gap-3 bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 cursor-pointer">
            <input
              type="checkbox"
              checked={adaptiveHalt}
              onChange={(e) => setAdaptiveHalt(e.target.checked)}
            />
            <span className="text-sm text-gray-300">
              Adaptive halting
            </span>
          </label>

          <label className="flex items-center gap-3 bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 cursor-pointer">
            <input
              type="checkbox"
              checked={rereadMemory}
              onChange={(e) => setRereadMemory(e.target.checked)}
            />
            <span className="text-sm text-gray-300">
              Re-read memory
            </span>
          </label>
        </div>

        <button
          onClick={handleRun}
          disabled={loading}
          className="mt-6 flex items-center justify-center gap-2 w-full md:w-auto px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 text-white rounded-lg font-semibold transition-colors"
        >
          {loading ? (
            <>
              <RefreshCw className="animate-spin" size={18} />
              Running recurrent computation...
            </>
          ) : (
            <>
              <Play size={18} />
              Run Experiment
            </>
          )}
        </button>
      </section>

      {/* Empty state */}
      {!data && !loading && (
        <section className="bg-gray-800/50 border border-dashed border-gray-700 rounded-xl p-12 text-center">
          <Activity className="mx-auto text-gray-600" size={40} />
          <h3 className="mt-4 text-lg font-semibold text-gray-400">
            Ready for an experiment
          </h3>
          <p className="text-sm text-gray-600 mt-2">
            Change the configuration above and run an episode.
          </p>
        </section>
      )}

      {/* Results */}
      {data && (
        <>
          {/* Result banner */}
          <section
            className={`rounded-xl border p-5 ${
              data.evaluation.exact_match
                ? 'bg-emerald-500/10 border-emerald-500/30'
                : 'bg-amber-500/10 border-amber-500/30'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-wider text-gray-500">
                  Evaluation Result
                </p>

                <h3
                  className={`text-2xl font-bold mt-1 ${
                    data.evaluation.exact_match
                      ? 'text-emerald-400'
                      : 'text-amber-400'
                  }`}
                >
                  {data.evaluation.exact_match
                    ? 'Exact Match'
                    : 'Partial Match'}
                </h3>
              </div>

              <div className="text-right">
                <p className="text-3xl font-mono text-gray-100">
                  {(data.evaluation.cell_accuracy * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-gray-500">
                  {data.evaluation.correct_cells}/
                  {data.evaluation.total_cells} cells correct
                </p>
              </div>
            </div>
          </section>

          {/* Demonstrations */}
          <section className="bg-gray-800 p-6 rounded-xl border border-gray-700">
            <div className="flex items-center gap-2 mb-5">
              <Database size={18} className="text-blue-400" />
              <h3 className="font-semibold text-gray-200">
                Demonstrations
              </h3>
            </div>

            <div className="flex flex-wrap gap-6">
              {data.task.demonstrations.map((demo, idx) => (
                <div
                  key={idx}
                  className="bg-gray-900 border border-gray-700 rounded-xl p-4"
                >
                  <p className="text-xs text-gray-500 mb-3">
                    Demonstration {idx + 1}
                  </p>

                  <div className="flex items-center gap-3">
                    <div>
                      <p className="text-[10px] text-gray-600 mb-2">INPUT</p>
                      {renderGrid(demo.input_grid, true)}
                    </div>

                    <span className="text-gray-500 text-xl">→</span>

                    <div>
                      <p className="text-[10px] text-gray-600 mb-2">OUTPUT</p>
                      {renderGrid(demo.output_grid, true)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Query / prediction / target */}
          <section className="bg-gray-800 p-6 rounded-xl border border-gray-700">
            <h3 className="font-semibold text-gray-200 mb-5">
              Query → Recurrent Computation → Prediction
            </h3>

            <div className="flex flex-wrap items-center justify-center gap-8">
              <div className="text-center">
                <p className="text-xs text-gray-500 mb-3">QUERY INPUT</p>
                {renderGrid(data.task.query.input_grid)}
              </div>

              <div className="text-center">
                <div className="text-blue-400 text-xs font-semibold">
                  RECURRENT
                </div>
                <div className="text-gray-500 text-2xl mt-1">→</div>
                <p className="text-[10px] text-gray-600 mt-1">
                  latent computation
                </p>
              </div>

              <div className="text-center">
                <p className="text-xs text-gray-500 mb-3">MODEL PREDICTION</p>
                {renderGrid(data.prediction)}
              </div>
            </div>
          </section>

          {/* Telemetry */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Activity size={18} className="text-blue-400" />
              <h3 className="font-semibold text-gray-200">
                Computation Telemetry
              </h3>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              <Metric
                label="Memory Norm"
                value={data.telemetry.memory.norm.toFixed(3)}
                description="State magnitude"
              />

              <Metric
                label="Effective Rank"
                value={data.telemetry.memory.effective_rank.toFixed(2)}
                description="Memory state complexity"
              />

              <Metric
                label="Entropy"
                value={data.telemetry.memory.entropy.toFixed(3)}
                description="Singular-value entropy"
              />

              <Metric
                label="Sparsity"
                value={`${(
                  data.telemetry.neurons.mean_sparsity * 100
                ).toFixed(1)}%`}
                description="Inactive neuron fraction"
              />

              <Metric
                label="Reasoning Steps"
                value={String(
                  data.telemetry.reasoning.trajectory_length
                )}
                description="Latent trajectory length"
              />
            </div>
          </section>

          {/* Configuration */}
          <section className="bg-gray-800 border border-gray-700 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Reproducibility
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-600 text-xs">Task</p>
                <p className="text-gray-300 mt-1">{data.task.family}</p>
              </div>

              <div>
                <p className="text-gray-600 text-xs">Seed</p>
                <p className="font-mono text-gray-300 mt-1">
                  {data.task.seed}
                </p>
              </div>

              <div>
                <p className="text-gray-600 text-xs">Memory</p>
                <p className="text-gray-300 mt-1">
                  {data.configuration.memory_mode}
                </p>
              </div>

              <div>
                <p className="text-gray-600 text-xs">Depth</p>
                <p className="font-mono text-gray-300 mt-1">
                  {data.configuration.reasoning_depth}
                </p>
              </div>
            </div>
          </section>

          {/* Scientific note */}
          <section className="text-xs text-gray-600 leading-relaxed border-t border-gray-800 pt-5">
            <strong className="text-gray-500">Interpretation note:</strong>{' '}
            results shown here are from the project's controlled experimental
            task generators. They are not official BDH or BDH-CQ benchmark
            results. The telemetry exposes aggregate diagnostics rather than
            a textual chain-of-thought.
          </section>
        </>
      )}
    </div>
  );
};

export default ARCPlayground;