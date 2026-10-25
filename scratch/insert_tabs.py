import re

file_path = 'frontend/src/pages/ValidationRunDetailPage.tsx'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        </div>
      </div>
    </div>
  );
}'''

tabs_code = '''        </div>
      </div>
    </div>
  )}

      {/* Tab 2: Dedicated Interactive Visual Analytics Engine */}
      {activeViewTab === 'visual_analytics' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-blue-600" />
                  <h3 className="text-base font-bold text-gray-900">
                    Interactive Visual Analytics Engine
                  </h3>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Empirical interactive charts generated dynamically from persisted validation run metrics. Explore distributions, threshold boundaries, calibration curves, and export underlying series to CSV.
                </p>
              </div>

              {/* Category Filter Pills */}
              <div className="flex flex-wrap gap-1.5">
                {[
                  { id: 'all', label: 'All Pillars' },
                  { id: 'performance', label: 'Performance' },
                  { id: 'data_quality', label: 'Data Quality' },
                  { id: 'drift', label: 'Drift' },
                  { id: 'fairness', label: 'Fairness' },
                  { id: 'robustness', label: 'Robustness' },
                  { id: 'explainability', label: 'Explainability' },
                  { id: 'reliability', label: 'Reliability' },
                ].map((pill) => (
                  <button
                    key={pill.id}
                    onClick={() => setSelectedChartCategory(pill.id)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                      selectedChartCategory === pill.id
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {pill.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {filteredCharts.length === 0 ? (
            <div className="bg-white rounded-lg border border-gray-200 p-12 text-center text-gray-500">
              <PieChart className="w-8 h-8 text-gray-400 mx-auto mb-2 opacity-60" />
              No visual analytics charts available for the selected pillar.
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {filteredCharts.map((chart) => (
                <ChartCard key={chart.chart_id} chart={chart} runId={run.id} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Dedicated Predictions Comparison */}
      {activeViewTab === 'predictions' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-bold text-gray-900">
                Row-Level Prediction Comparison Table
              </h3>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Sample-by-sample model predictions compared directly against ground truth labels (Deepchecks standard).
            </p>
          </div>
          {perfMetrics.predictions?.sample_rows ? (
            <div className="border border-gray-200 rounded-lg p-4 bg-white shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h4 className="text-sm font-semibold text-gray-800">
                    Row-Level Predictions
                  </h4>
                  <p className="text-xs text-gray-500">
                    {perfMetrics.predictions.total_rows} evaluated, {perfMetrics.predictions.incorrect_rows ?? 0} misclassified
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="inline-flex rounded-md shadow-xs" role="group">
                    <button
                      type="button"
                      onClick={() => { setPredictionFilter('all'); setPredictionPage(1); }}
                      className={`px-2.5 py-1 text-xs font-medium rounded-l-md border ${
                        predictionFilter === 'all'
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      All ({perfMetrics.predictions.sample_rows.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => { setPredictionFilter('incorrect'); setPredictionPage(1); }}
                      className={`px-2.5 py-1 text-xs font-medium border-t border-b ${
                        predictionFilter === 'incorrect'
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      Misclassified ({perfMetrics.predictions.sample_rows.filter((r: any) => !r.correct).length})
                    </button>
                    <button
                      type="button"
                      onClick={() => { setPredictionFilter('correct'); setPredictionPage(1); }}
                      className={`px-2.5 py-1 text-xs font-medium rounded-r-md border ${
                        predictionFilter === 'correct'
                          ? 'bg-blue-600 text-white border-blue-600'
                          : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      Correct ({perfMetrics.predictions.sample_rows.filter((r: any) => r.correct).length})
                    </button>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto max-h-96">
                <table className="min-w-full text-xs divide-y divide-gray-200">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Row #</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Actual Label</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Predicted</th>
                      <th className="px-3 py-2 text-center font-medium text-gray-600">Verdict</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Prediction Probabilities</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 bg-white">
                    {perfMetrics.predictions.sample_rows
                      .filter((row: any) => {
                        if (predictionFilter === 'incorrect') return !row.correct;
                        if (predictionFilter === 'correct') return row.correct;
                        return true;
                      })
                      .map((row: any) => (
                        <tr key={row.row_index} className={row.correct ? 'hover:bg-gray-50' : 'bg-red-50/40 hover:bg-red-50'}>
                          <td className="px-3 py-2 text-gray-500 font-mono">#{row.row_index}</td>
                          <td className="px-3 py-2 font-medium text-gray-900">{String(row.actual)}</td>
                          <td className="px-3 py-2 font-semibold text-gray-900">{String(row.predicted)}</td>
                          <td className="px-3 py-2 text-center">
                            {row.correct ? (
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-green-100 text-green-800">
                                Match
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-red-100 text-red-800">
                                Error
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2 font-mono text-[11px]">
                            {row.probabilities ? (
                              <div className="flex gap-2">
                                {Object.entries(row.probabilities).map(([k, v]: [string, any]) => (
                                  <span key={k} className="text-gray-600">
                                    {k}: <span className="font-semibold text-gray-900">{((v as number) * 100).toFixed(1)}%</span>
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className="text-gray-400 italic">Deterministic Output (no predict_proba)</span>
                            )}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-500 text-sm">
              No row-level prediction comparison data logged for this run.
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Dedicated Findings */}
      {activeViewTab === 'findings' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-500" />
              <h3 className="text-base font-semibold text-gray-900">Validation Findings & Recommendations</h3>
              <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full font-medium">
                {filteredFindings.length}
              </span>
            </div>

            <div className="flex flex-wrap gap-1 text-xs">
              {['all', 'critical', 'high', 'medium', 'low', 'info'].map((sev) => (
                <button
                  key={sev}
                  onClick={() => setSelectedSeverity(sev)}
                  className={`px-2.5 py-1 rounded-md uppercase font-medium transition-colors ${
                    selectedSeverity === sev
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {sev}
                </button>
              ))}
            </div>
          </div>

          <div className="p-6">
            {filteredFindings.length === 0 ? (
              <div className="text-center py-8 text-gray-500 text-sm">
                <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2 opacity-60" />
                No findings matching the selected filter.
              </div>
            ) : (
              <div className="space-y-4">
                {filteredFindings.map((finding) => (
                  <div
                    key={finding.id}
                    className="p-4 rounded-lg border border-gray-200 bg-white hover:border-gray-300 transition-shadow shadow-xs space-y-2"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <SeverityBadge severity={finding.severity} />
                          <span className="text-xs font-semibold uppercase text-gray-400 tracking-wider">
                            {finding.category}
                          </span>
                        </div>
                        <h4 className="text-sm font-semibold text-gray-900 mt-1">{finding.title}</h4>
                      </div>
                      {finding.validator_name && (
                        <span className="text-[11px] text-gray-400 bg-gray-50 px-2 py-0.5 rounded border border-gray-100">
                          {finding.validator_name} v{finding.validator_version}
                        </span>
                      )}
                    </div>

                    <p className="text-xs text-gray-600 leading-relaxed">{finding.description}</p>

                    {(finding.metric_name || finding.threshold) && (
                      <div className="flex flex-wrap gap-4 text-xs bg-gray-50 p-2.5 rounded border border-gray-100 font-mono">
                        {finding.metric_name && (
                          <div>
                            <span className="text-gray-500 font-sans">Observed Metric ({finding.metric_name}): </span>
                            <span className="font-bold text-gray-800">{finding.metric_value}</span>
                          </div>
                        )}
                        {finding.threshold && (
                          <div>
                            <span className="text-gray-500 font-sans">Acceptance Threshold: </span>
                            <span className="text-gray-700">{finding.threshold}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {finding.recommendation && (
                      <div className="text-xs text-blue-900 bg-blue-50/80 p-2.5 rounded border border-blue-100/70">
                        <span className="font-semibold text-blue-800">Recommendation: </span>
                        {finding.recommendation}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}'''

# Normalize newlines
content_norm = content.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')

if target_norm in content_norm:
    new_content = content_norm.replace(target_norm, tabs_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS: Tabs inserted successfully.")
else:
    print("ERROR: Target string not found in content.")
