import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import ScoreCard from '../components/ScoreCard';
import DataTable from '../components/DataTable';
import StatusBadge from '../components/StatusBadge';
import SeverityBadge from '../components/SeverityBadge';
import api from '../services/api';
import { ValidationRun, Finding } from '../types';

export default function DashboardPage() {
  const [runs, setRuns] = useState<ValidationRun[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.get<ValidationRun[]>('/validation/runs').catch(() => ({ data: [] })),
      api.get<{ findings: Finding[]; total: number }>('/validation/findings?limit=5').catch(() => ({ data: { findings: [] } })),
    ])
      .then(([runsRes, findingsRes]) => {
        setRuns(runsRes.data || []);
        setFindings(findingsRes.data?.findings || []);
      })
      .finally(() => setLoading(false));
  }, []);

  // Compute latest scores or averages across runs
  const latestRun = runs.length > 0 ? runs[0] : null;
  const overallScore = latestRun?.overall_score ?? null;
  const categoryScores = latestRun?.category_scores_json || {};

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-2">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Platform Model Health Dashboard</h1>
          <p className="text-gray-500 text-sm">Real-time assurance and multi-dimensional evaluation metrics</p>
        </div>
        <Link
          to="/validation-runs"
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors shadow-xs"
        >
          New Validation Run
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="md:col-span-1">
          <ScoreCard label="Platform Model Health Score" score={overallScore} />
        </div>
        <div className="md:col-span-3 grid grid-cols-2 sm:grid-cols-3 gap-3">
          <ScoreCard label="Performance" score={categoryScores.performance ?? null} />
          <ScoreCard label="Data Quality" score={categoryScores.data_quality ?? null} />
          <ScoreCard label="Fairness" score={categoryScores.fairness ?? null} />
          <ScoreCard label="Drift" score={categoryScores.drift ?? null} />
          <ScoreCard label="Robustness" score={categoryScores.robustness ?? null} />
          <ScoreCard label="Explainability" score={categoryScores.explainability ?? null} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
            <h3 className="text-base font-semibold text-gray-900">Recent Validation Runs</h3>
            <Link to="/validation-runs" className="text-xs text-blue-600 hover:text-blue-800 font-medium">
              View All ({runs.length})
            </Link>
          </div>
          <div className="p-4">
            <DataTable<ValidationRun>
              columns={[
                { key: 'run_number', header: 'Run ID', render: (r) => (
                  <Link to={`/validation-runs/${r.id}`} className="font-semibold text-blue-600 hover:underline">
                    {r.run_number}
                  </Link>
                )},
                { key: 'status', header: 'Status', render: (item) => <StatusBadge status={item.status} /> },
                { key: 'overall_score', header: 'Score', render: (r) => r.overall_score !== null ? `${Math.round(r.overall_score)}/100` : '-' },
                { key: 'started_at', header: 'Date', render: (r) => r.started_at ? new Date(r.started_at).toLocaleDateString() : '-' },
              ]}
              data={runs.slice(0, 5)}
              emptyMessage={loading ? "Loading runs..." : "No recent validation runs."}
            />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
            <h3 className="text-base font-semibold text-gray-900">Critical & High Findings</h3>
            <Link to="/findings" className="text-xs text-blue-600 hover:text-blue-800 font-medium">
              View All
            </Link>
          </div>
          <div className="p-4">
            <DataTable<Finding>
              columns={[
                { key: 'severity', header: 'Severity', render: (item) => <SeverityBadge severity={item.severity} /> },
                { key: 'category', header: 'Category' },
                { key: 'title', header: 'Title', render: (f) => (
                  <div className="truncate max-w-xs font-medium text-gray-800" title={f.title}>
                    {f.title}
                  </div>
                )},
              ]}
              data={findings}
              emptyMessage={loading ? "Loading findings..." : "No findings recorded."}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
