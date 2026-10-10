import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload, CheckCircle, AlertTriangle, XCircle, Info, RefreshCw, FileText,
  Play, Plus, ArrowRight, ShieldCheck, Database, Cpu, HelpCircle, Eye,
  Sparkles, Check, ChevronDown, Activity, AlertOctagon, Layers
} from 'lucide-react';
import api from '../services/api';
import DataTable from '../components/DataTable';
import StatusBadge from '../components/StatusBadge';
import { ValidationRun, Project, MLModel, Dataset, CompatibilityCheckResponse } from '../types';

interface ValidatorCard {
  id: string;
  name: string;
  description: string;
  requiresSensitive: boolean;
  requiresReference: boolean;
  requiresProba: boolean;
}

const VALIDATORS: ValidatorCard[] = [
  {
    id: 'performance',
    name: 'Performance',
    description: 'Empirical classification/regression scoring (Accuracy, ROC-AUC, F1, Log Loss, MSE)',
    requiresSensitive: false,
    requiresReference: false,
    requiresProba: false,
  },
  {
    id: 'data_quality',
    name: 'Data Quality & Integrity',
    description: 'Validates missing ratios, distributions, duplicate contamination, and schema parity',
    requiresSensitive: false,
    requiresReference: false,
    requiresProba: false,
  },
  {
    id: 'fairness',
    name: 'Fairness & Algorithmic Bias',
    description: 'Audits demographic parity & EEOC 80% four-fifths rule across protected groups',
    requiresSensitive: true,
    requiresReference: false,
    requiresProba: false,
  },
  {
    id: 'drift',
    name: 'Data & Prediction Drift',
    description: 'Calculates population stability (PSI) and Kolmogorov-Smirnov distribution shifts',
    requiresSensitive: false,
    requiresReference: true,
    requiresProba: false,
  },
  {
    id: 'robustness',
    name: 'Adversarial Robustness',
    description: 'Tests resilience against gaussian feature perturbations and input noise',
    requiresSensitive: false,
    requiresReference: false,
    requiresProba: false,
  },
  {
    id: 'explainability',
    name: 'Explainability & Attribution',
    description: 'Analyzes feature attribution balance and checks for single-feature dominance',
    requiresSensitive: false,
    requiresReference: false,
    requiresProba: false,
  },
  {
    id: 'reliability',
    name: 'Reliability & Calibration',
    description: 'Measures expected calibration error (ECE) and probability reliability',
    requiresSensitive: false,
    requiresReference: false,
    requiresProba: true,
  },
];

export default function ValidationRunsPage() {
  const [runs, setRuns] = useState<ValidationRun[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [models, setModels] = useState<MLModel[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  // Form State
  const [showConfig, setShowConfig] = useState(true);

  // Step 1: Project Context State (A: Select Existing, B: Create New, C: Quick Validation)
  const [projectMode, setProjectMode] = useState<'existing' | 'new' | 'quick'>('existing');
  const [selectedProject, setSelectedProject] = useState('');
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);
  const [quickProjectObj, setQuickProjectObj] = useState<Project | null>(null);

  // Step 2: Model selection / upload
  const [selectedModel, setSelectedModel] = useState('');
  const [showModelUpload, setShowModelUpload] = useState(false);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelDragActive, setModelDragActive] = useState(false);
  const [modelUploading, setModelUploading] = useState(false);
  const [modelUploadError, setModelUploadError] = useState<string | null>(null);

  // Step 3: Dataset selection / upload
  const [selectedDataset, setSelectedDataset] = useState('');
  const [showDatasetUpload, setShowDatasetUpload] = useState(false);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetDragActive, setDatasetDragActive] = useState(false);
  const [datasetUploading, setDatasetUploading] = useState(false);

  // Step 4: Automatic Model & Dataset Inspection Analysis State
  const [datasetPreview, setDatasetPreview] = useState<any | null>(null);
  const [modelInspect, setModelInspect] = useState<any | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  // Step 5: Selected validators
  const [selectedValidators, setSelectedValidators] = useState<string[]>([
    'performance', 'data_quality', 'fairness', 'drift', 'robustness', 'explainability', 'reliability'
  ]);

  // Compatibility checking
  const [checkingCompat, setCheckingCompat] = useState(false);
  const [compatResult, setCompatResult] = useState<CompatibilityCheckResponse | null>(null);
  const [compatError, setCompatError] = useState<string | null>(null);

  // Execution state
  const [startingRun, setStartingRun] = useState(false);
  const [runSuccess, setRunSuccess] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const navigate = useNavigate();

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const [rRes, pRes, mRes, dRes] = await Promise.all([
        api.get<ValidationRun[]>('/validation/runs'),
        api.get<Project[]>('/projects'),
        api.get<MLModel[]>('/models'),
        api.get<Dataset[]>('/datasets')
      ]);
      setRuns(rRes.data);
      setProjects(pRes.data);
      setModels(mRes.data);
      setDatasets(dRes.data);
      if (pRes.data.length > 0 && !selectedProject) {
        setSelectedProject(pRes.data[0].id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRuns = async () => {
    try {
      const { data } = await api.get<ValidationRun[]>('/validation/runs');
      setRuns(data);
    } catch (err) {
      console.error(err);
    }
  };

  // Helper to ensure Quick Validation workspace project exists
  const ensureQuickProject = async (): Promise<string> => {
    if (quickProjectObj) return quickProjectObj.id;
    const existing = projects.find(p => p.name === 'Quick Validation Workspace');
    if (existing) {
      setQuickProjectObj(existing);
      return existing.id;
    }
    try {
      const { data } = await api.post<Project>('/projects', {
        name: 'Quick Validation Workspace',
        description: 'Automatic ad-hoc validation workspace'
      });
      setProjects(prev => [data, ...prev]);
      setQuickProjectObj(data);
      return data.id;
    } catch (e) {
      return projects.length > 0 ? projects[0].id : '';
    }
  };

  // Step 1: Create New Project Handler
  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;
    setCreatingProject(true);
    try {
      const { data } = await api.post<Project>('/projects', {
        name: newProjectName.trim(),
        description: newProjectDesc.trim() || undefined
      });
      setProjects(prev => [data, ...prev]);
      setSelectedProject(data.id);
      setProjectMode('existing');
      setNewProjectName('');
      setNewProjectDesc('');
      setCompatResult(null);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to create project.');
    } finally {
      setCreatingProject(false);
    }
  };

  // Auto-analysis of model & dataset
  useEffect(() => {
    if (!selectedModel) {
      setModelInspect(null);
    } else {
      api.get(`/models/${selectedModel}/inspect`)
        .then(res => setModelInspect(res.data))
        .catch(() => setModelInspect(null));
    }
  }, [selectedModel]);

  useEffect(() => {
    if (!selectedDataset) {
      setDatasetPreview(null);
      setCompatResult(null);
    } else {
      setLoadingAnalysis(true);
      api.get(`/datasets/${selectedDataset}/preview?max_rows=15`)
        .then(res => setDatasetPreview(res.data))
        .catch(() => setDatasetPreview(null))
        .finally(() => setLoadingAnalysis(false));
      setCompatResult(null);
    }
  }, [selectedDataset]);

  // Model upload handler
  const handleModelUpload = async () => {
    if (!modelFile) return;
    setModelUploading(true);
    setModelUploadError(null);
    try {
      let activeProjectId = selectedProject;
      if (projectMode === 'quick' || !activeProjectId) {
        activeProjectId = await ensureQuickProject();
        setSelectedProject(activeProjectId);
      }
      const formData = new FormData();
      formData.append('project_id', activeProjectId);
      formData.append('name', modelFile.name.replace(/\.[^/.]+$/, ''));
      formData.append('file', modelFile);
      const res = await api.post<MLModel>('/models/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setModels(prev => [res.data, ...prev]);
      setSelectedModel(res.data.id);
      setShowModelUpload(false);
      setModelFile(null);
      setModelUploadError(null);
      setCompatResult(null);
    } catch (err: any) {
      const errDetail = err.response?.data?.detail || err.message || 'Failed to upload model file.';
      setModelUploadError(errDetail);
    } finally {
      setModelUploading(false);
    }
  };

  // Dataset upload handler
  const handleDatasetUpload = async () => {
    if (!datasetFile) return;
    setDatasetUploading(true);
    try {
      let activeProjectId = selectedProject;
      if (projectMode === 'quick' || !activeProjectId) {
        activeProjectId = await ensureQuickProject();
        setSelectedProject(activeProjectId);
      }
      const formData = new FormData();
      formData.append('project_id', activeProjectId);
      formData.append('name', datasetFile.name.replace(/\.[^/.]+$/, ''));
      formData.append('file', datasetFile);
      const res = await api.post<Dataset>('/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDatasets(prev => [res.data, ...prev]);
      setSelectedDataset(res.data.id);
      setShowDatasetUpload(false);
      setDatasetFile(null);
      setCompatResult(null);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to upload dataset CSV.');
    } finally {
      setDatasetUploading(false);
    }
  };

  // Pre-flight compatibility check handler (Step 5)
  const handleCheckCompatibility = async () => {
    if (!selectedModel || !selectedDataset) {
      alert('Please select or upload a Model and Test Dataset first.');
      return;
    }
    setCheckingCompat(true);
    setCompatError(null);
    setCompatResult(null);
    try {
      const res = await api.post<CompatibilityCheckResponse>('/validation/compatibility-check', {
        model_id: selectedModel,
        dataset_id: selectedDataset,
        selected_validators: selectedValidators,
      });
      setCompatResult(res.data);
    } catch (err: any) {
      setCompatError(err.response?.data?.detail || 'Compatibility check failed.');
    } finally {
      setCheckingCompat(false);
    }
  };

  // Start Full Validation handler (Step 6)
  const handleStartValidation = async () => {
    if (!selectedModel || !selectedDataset) {
      setRunError('Validation could not start: Both Model and Test Dataset must be selected.');
      return;
    }
    setStartingRun(true);
    setRunSuccess(null);
    setRunError(null);
    try {
      let activeProjectId = selectedProject;
      if (projectMode === 'quick' || !activeProjectId) {
        activeProjectId = await ensureQuickProject();
      }
      const payload = {
        model_id: selectedModel,
        dataset_id: selectedDataset,
        selected_validators: selectedValidators,
        config: {
          test_suite_version: '2026.1.0',
          execution_trigger: 'automatic_complete_audit',
        }
      };
      const res = await api.post<ValidationRun>('/validation/runs', payload);
      setRunSuccess(`Full Validation Run ${res.data.run_number} completed successfully!`);
      await fetchRuns();
      setTimeout(() => {
        navigate(`/validation-runs/${res.data.id}`);
      }, 1000);
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Execution failed';
      setRunError(`Validation could not start: ${detail}`);
    } finally {
      setStartingRun(false);
    }
  };

  const toggleValidator = (id: string) => {
    setSelectedValidators(prev =>
      prev.includes(id) ? prev.filter(v => v !== id) : [...prev, id]
    );
    setCompatResult(null);
  };

  const projectModels = projectMode === 'quick' 
    ? models 
    : models.filter(m => m.project_id === selectedProject);

  const projectDatasets = projectMode === 'quick'
    ? datasets
    : datasets.filter(d => d.project_id === selectedProject);

  const currentModelObj = models.find(m => m.id === selectedModel);
  const currentDatasetObj = datasets.find(d => d.id === selectedDataset);
  const isStep1Ready = projectMode === 'quick' || !!selectedProject;

  return (
    <div className="space-y-8 pb-16">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Validation Runs</h1>
          <p className="text-gray-500">Automated end-to-end model validation, statistical inspection, and assurance auditing</p>
        </div>
        <button
          onClick={() => setShowConfig(!showConfig)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            showConfig ? 'bg-gray-100 text-gray-700 hover:bg-gray-200' : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {showConfig ? 'Hide Configuration' : '+ New Validation Run'}
        </button>
      </div>

      {runSuccess && (
        <div className="bg-green-50 border border-green-200 p-4 rounded-lg flex items-center text-green-800">
          <CheckCircle className="h-5 w-5 mr-3 text-green-600 flex-shrink-0" />
          <span className="font-medium">{runSuccess} Redirecting to audit results...</span>
        </div>
      )}

      {/* Main Validation Setup Card */}
      {showConfig && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="bg-slate-50 border-b border-gray-200 px-6 py-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ShieldCheck className="h-5 w-5 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900">Start New Validation Run</h2>
            </div>
            <span className="text-xs font-mono uppercase bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-medium">
              Automated 6-Step Workflow
            </span>
          </div>

          <div className="p-6 space-y-8">
            {/* STEP 1 — PROJECT CONTEXT */}
            <div className="border-b border-gray-100 pb-6">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-semibold text-gray-800 flex items-center">
                  <span className="bg-blue-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center mr-2">1</span>
                  Project Context
                </label>
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => setProjectMode('existing')}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      projectMode === 'existing'
                        ? 'bg-blue-600 text-white shadow-sm'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    Select Existing Project
                  </button>
                  <button
                    type="button"
                    onClick={() => setProjectMode('new')}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      projectMode === 'new'
                        ? 'bg-blue-600 text-white shadow-sm'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    + Create New Project
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setProjectMode('quick');
                      setSelectedProject('');
                    }}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      projectMode === 'quick'
                        ? 'bg-emerald-600 text-white shadow-sm'
                        : 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                    }`}
                  >
                    ⚡ Quick Validation — No Project
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-500 mb-4">
                {projectMode === 'existing' && 'Select an existing governance project to associate this validation run.'}
                {projectMode === 'new' && 'Create a new project workspace directly into the database.'}
                {projectMode === 'quick' && 'Execute an immediate ad-hoc validation run across any uploaded or existing model and dataset without binding to a project.'}
              </p>

              {/* Option A: Select Existing Project */}
              {projectMode === 'existing' && (
                <div className="max-w-xl">
                  <select
                    value={selectedProject}
                    onChange={(e) => {
                      setSelectedProject(e.target.value);
                      setSelectedModel('');
                      setSelectedDataset('');
                      setCompatResult(null);
                    }}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                  >
                    <option value="">Select Project...</option>
                    {projects.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Option B: Create New Project */}
              {projectMode === 'new' && (
                <form onSubmit={handleCreateProject} className="max-w-xl bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1">
                      Project Name <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      placeholder="e.g. Credit Risk Assessment 2026"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-xs bg-white focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1">
                      Description <span className="text-gray-400 font-normal">(optional)</span>
                    </label>
                    <textarea
                      rows={2}
                      placeholder="Scope, validation objectives, or notes..."
                      value={newProjectDesc}
                      onChange={(e) => setNewProjectDesc(e.target.value)}
                      className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-xs bg-white focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div className="flex justify-end space-x-2 pt-1">
                    <button
                      type="button"
                      onClick={() => setProjectMode('existing')}
                      className="px-3 py-1.5 rounded text-xs text-gray-600 hover:text-gray-800"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={creatingProject || !newProjectName.trim()}
                      className="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-xs font-semibold hover:bg-blue-700 disabled:opacity-50"
                    >
                      {creatingProject ? 'Creating Project...' : 'Create Project & Continue'}
                    </button>
                  </div>
                </form>
              )}

              {/* Option C: Quick Validation Banner */}
              {projectMode === 'quick' && (
                <div className="max-w-xl bg-emerald-50 border border-emerald-200 rounded-xl p-3.5 flex items-center justify-between text-xs text-emerald-900">
                  <div className="flex items-center space-x-2">
                    <Sparkles className="h-4 w-4 text-emerald-600 flex-shrink-0" />
                    <span><strong>Quick Validation Active:</strong> Upload or pick any model and test dataset below to run immediate validation.</span>
                  </div>
                  <span className="bg-emerald-200/70 text-emerald-800 font-semibold px-2 py-0.5 rounded text-[11px]">Unrestricted</span>
                </div>
              )}
            </div>

            {isStep1Ready ? (
              <>
                {/* STEP 2 — MODEL UNDER TEST */}
                <div className="border-b border-gray-100 pb-6">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-semibold text-gray-800 flex items-center">
                      <span className="bg-blue-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center mr-2">2</span>
                      Model Under Test
                    </label>
                    <button
                      type="button"
                      onClick={() => setShowModelUpload(!showModelUpload)}
                      className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center"
                    >
                      {showModelUpload ? 'Choose Existing Model' : '+ Upload New Model'}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mb-3">
                    Trained model artifact (.pkl, .joblib, .pickle, .keras, .h5, .onnx, max 100MB). Supports Scikit-Learn pipelines and TensorFlow/Keras neural networks.
                  </p>

                  {!showModelUpload ? (
                    <div className="max-w-xl space-y-2">
                      <select
                        value={selectedModel}
                        onChange={(e) => {
                          setSelectedModel(e.target.value);
                          setCompatResult(null);
                        }}
                        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                      >
                        <option value="">Select model...</option>
                        {projectModels.map(m => (
                          <option key={m.id} value={m.id}>
                            {m.name} ({m.framework || 'Python'} - {m.model_type || 'Estimator'})
                          </option>
                        ))}
                      </select>
                      {currentModelObj && (
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-700 flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Cpu className="h-4 w-4 text-slate-500" />
                            <div>
                              <p className="font-semibold text-gray-900">{currentModelObj.name}</p>
                              <p className="text-gray-500">{currentModelObj.filename} • {(currentModelObj.file_size / 1024).toFixed(1)} KB • {currentModelObj.framework || 'scikit-learn'}</p>
                            </div>
                          </div>
                          <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded text-xs font-medium">✓ Ready</span>
                        </div>
                      )}
                    </div>
                  ) : (
                    /* Drag & Drop Model Area */
                    <div className="max-w-xl space-y-3">
                      {modelUploadError && (
                        <div className="bg-red-50 border border-red-200 p-3 rounded-lg text-xs text-red-800 flex items-start">
                          <XCircle className="h-4 w-4 text-red-600 mr-2 mt-0.5 flex-shrink-0" />
                          <div>
                            <p className="font-semibold">Model Upload Failed</p>
                            <p className="mt-0.5">{modelUploadError}</p>
                          </div>
                        </div>
                      )}
                      <div
                        onDragOver={(e) => { e.preventDefault(); setModelDragActive(true); }}
                        onDragLeave={() => setModelDragActive(false)}
                        onDrop={(e) => {
                          e.preventDefault();
                          setModelDragActive(false);
                          setModelUploadError(null);
                          if (e.dataTransfer.files?.[0]) setModelFile(e.dataTransfer.files[0]);
                        }}
                        className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer ${
                          modelDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400 bg-gray-50/50'
                        }`}
                      >
                        <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                        <p className="text-sm font-medium text-gray-700">Drag & Drop ML Model Here</p>
                        <p className="text-xs text-gray-500 mt-1">Accepts .pkl, .joblib, .pickle, .keras, .h5, .onnx files up to 100MB</p>
                        <label className="mt-3 inline-block bg-white border border-gray-300 text-gray-700 px-3 py-1.5 rounded-md text-xs font-medium hover:bg-gray-50 cursor-pointer shadow-sm">
                          Choose Model File
                          <input
                            type="file"
                            accept=".pkl,.joblib,.pickle,.keras,.h5,.onnx"
                            onChange={(e) => {
                              setModelUploadError(null);
                              if (e.target.files?.[0]) setModelFile(e.target.files[0]);
                            }}
                            className="hidden"
                          />
                        </label>
                      </div>

                      {modelFile && (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs flex items-center justify-between">
                          <div>
                            <p className="font-semibold text-blue-900">✓ Model file selected: {modelFile.name}</p>
                            <p className="text-blue-700">Size: {(modelFile.size / 1024).toFixed(1)} KB</p>
                          </div>
                          <div className="flex space-x-2">
                            <button
                              type="button"
                              onClick={() => {
                                setModelFile(null);
                                setModelUploadError(null);
                              }}
                              className="text-gray-500 hover:text-gray-700 font-medium px-2 py-1"
                            >
                              Remove
                            </button>
                            <button
                              type="button"
                              onClick={handleModelUpload}
                              disabled={modelUploading}
                              className="bg-blue-600 text-white px-3 py-1 rounded font-medium hover:bg-blue-700 disabled:opacity-50"
                            >
                              {modelUploading ? 'Uploading...' : 'Upload & Inspect'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* STEP 3 — TEST DATASET */}
                <div className="border-b border-gray-100 pb-6">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-semibold text-gray-800 flex items-center">
                      <span className="bg-blue-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center mr-2">3</span>
                      Test Dataset
                    </label>
                    <button
                      type="button"
                      onClick={() => setShowDatasetUpload(!showDatasetUpload)}
                      className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center"
                    >
                      {showDatasetUpload ? 'Choose Existing Dataset' : '+ Upload New Dataset'}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mb-3">Structured evaluation dataset in CSV format containing feature inputs and ground truth.</p>

                  {!showDatasetUpload ? (
                    <div className="max-w-xl space-y-2">
                      <select
                        value={selectedDataset}
                        onChange={(e) => {
                          setSelectedDataset(e.target.value);
                          setCompatResult(null);
                        }}
                        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                      >
                        <option value="">Select test dataset...</option>
                        {projectDatasets.map(d => (
                          <option key={d.id} value={d.id}>
                            {d.name} ({d.row_count ? `${d.row_count.toLocaleString()} rows` : d.filename})
                          </option>
                        ))}
                      </select>
                      {currentDatasetObj && (
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-700 flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Database className="h-4 w-4 text-slate-500" />
                            <div>
                              <p className="font-semibold text-gray-900">{currentDatasetObj.name}</p>
                              <p className="text-gray-500">
                                {currentDatasetObj.filename} • {currentDatasetObj.row_count?.toLocaleString()} rows • {currentDatasetObj.column_count} columns
                              </p>
                            </div>
                          </div>
                          <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded text-xs font-medium">✓ Loaded</span>
                        </div>
                      )}
                    </div>
                  ) : (
                    /* Drag & Drop Dataset Area */
                    <div className="max-w-xl space-y-3">
                      <div
                        onDragOver={(e) => { e.preventDefault(); setDatasetDragActive(true); }}
                        onDragLeave={() => setDatasetDragActive(false)}
                        onDrop={(e) => {
                          e.preventDefault();
                          setDatasetDragActive(false);
                          if (e.dataTransfer.files?.[0]) setDatasetFile(e.dataTransfer.files[0]);
                        }}
                        className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer ${
                          datasetDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400 bg-gray-50/50'
                        }`}
                      >
                        <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                        <p className="text-sm font-medium text-gray-700">Drag & Drop Test Dataset (CSV) Here</p>
                        <p className="text-xs text-gray-500 mt-1">Accepts .csv tabular files up to 100MB</p>
                        <label className="mt-3 inline-block bg-white border border-gray-300 text-gray-700 px-3 py-1.5 rounded-md text-xs font-medium hover:bg-gray-50 cursor-pointer shadow-sm">
                          Choose CSV File
                          <input
                            type="file"
                            accept=".csv"
                            onChange={(e) => e.target.files?.[0] && setDatasetFile(e.target.files[0])}
                            className="hidden"
                          />
                        </label>
                      </div>

                      {datasetFile && (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs flex items-center justify-between">
                          <div>
                            <p className="font-semibold text-blue-900">✓ CSV selected: {datasetFile.name}</p>
                            <p className="text-blue-700">Size: {(datasetFile.size / 1024).toFixed(1)} KB</p>
                          </div>
                          <div className="flex space-x-2">
                            <button
                              type="button"
                              onClick={() => setDatasetFile(null)}
                              className="text-gray-500 hover:text-gray-700 font-medium px-2 py-1"
                            >
                              Remove
                            </button>
                            <button
                              type="button"
                              onClick={handleDatasetUpload}
                              disabled={datasetUploading}
                              className="bg-blue-600 text-white px-3 py-1 rounded font-medium hover:bg-blue-700 disabled:opacity-50"
                            >
                              {datasetUploading ? 'Uploading...' : 'Upload & Parse'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* STEP 4 — AUTOMATIC MODEL & DATASET ANALYSIS */}
                <div className="border-b border-gray-100 pb-6">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-semibold text-gray-800 flex items-center">
                      <span className="bg-blue-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center mr-2">4</span>
                      Automatic Model & Dataset Analysis
                    </label>
                    <span className="text-xs text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-medium border border-emerald-200 flex items-center">
                      <Sparkles className="h-3 w-3 mr-1" /> Automated Full-Spectrum Inspection
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mb-4">
                    The platform automatically audits complete model architecture, expected input dimensions, dataset columns, missing rates, duplicate rows, infinite values, outliers, and class distributions.
                  </p>

                  {loadingAnalysis && (
                    <div className="p-6 text-center text-xs text-gray-500 bg-slate-50 rounded-xl border border-dashed border-gray-300">
                      <RefreshCw className="h-4 w-4 animate-spin mx-auto mb-2 text-blue-600" />
                      Analyzing model architecture and profiling dataset properties across all rows and columns...
                    </div>
                  )}

                  {!loadingAnalysis && (modelInspect || datasetPreview) ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {/* Model Architecture Analysis Card */}
                      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs space-y-3">
                        <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                          <span className="font-semibold text-gray-900 flex items-center">
                            <Cpu className="h-4 w-4 mr-1.5 text-blue-600" /> Model Architecture
                          </span>
                          <span className="text-[11px] font-mono text-gray-500">{modelInspect?.framework || 'scikit-learn'}</span>
                        </div>
                        {modelInspect ? (
                          <div className="space-y-2">
                            <div className="grid grid-cols-2 gap-2 text-gray-700">
                              <div><span className="text-gray-500">Estimator:</span> <strong className="text-gray-900">{modelInspect.model_class || 'Classifier'}</strong></div>
                              <div><span className="text-gray-500">Task Type:</span> <strong className="text-gray-900 capitalize">{modelInspect.model_type || 'Classification'}</strong></div>
                              <div><span className="text-gray-500">Expected Features:</span> <strong className="text-gray-900">{modelInspect.feature_count ?? (modelInspect.input_shape ? modelInspect.input_shape : 'Dynamic')}</strong></div>
                              <div><span className="text-gray-500">Probability Outputs:</span> <strong className={modelInspect.has_predict_proba ? 'text-green-700' : 'text-amber-700'}>{modelInspect.has_predict_proba ? 'Supported' : 'Not Supported'}</strong></div>
                              {modelInspect.total_parameters !== undefined && (
                                <div><span className="text-gray-500">Total Parameters:</span> <strong className="text-gray-900">{modelInspect.total_parameters.toLocaleString()}</strong></div>
                              )}
                              {modelInspect.layer_count !== undefined && (
                                <div><span className="text-gray-500">Layers:</span> <strong className="text-gray-900">{modelInspect.layer_count}</strong></div>
                              )}
                              {modelInspect.input_shape && (
                                <div className="col-span-2"><span className="text-gray-500">Input Shape:</span> <strong className="text-gray-900 font-mono text-[11px]">{modelInspect.input_shape}</strong></div>
                              )}
                            </div>
                            {modelInspect.feature_names && modelInspect.feature_names.length > 0 && (
                              <div className="pt-2 border-t border-slate-200">
                                <span className="text-gray-500 block mb-1">Expected Feature Names ({modelInspect.feature_names.length}):</span>
                                <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                                  {modelInspect.feature_names.map((f: string) => (
                                    <span key={f} className="bg-white border border-gray-200 px-1.5 py-0.5 rounded text-[10px] font-mono text-gray-800">
                                      {f}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-400">Select or upload a model above to view architecture inspection.</p>
                        )}
                      </div>

                      {/* Complete Dataset Quality & Health Card */}
                      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs space-y-3">
                        <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                          <span className="font-semibold text-gray-900 flex items-center">
                            <Database className="h-4 w-4 mr-1.5 text-blue-600" /> Complete Dataset Health Profile
                          </span>
                          <span className="text-[11px] font-mono text-gray-500">
                            {datasetPreview?.total_rows?.toLocaleString() ?? 0} rows • {datasetPreview?.total_columns ?? 0} cols
                          </span>
                        </div>
                        {datasetPreview ? (
                          <div className="space-y-2">
                            <div className="grid grid-cols-2 gap-2 text-gray-700">
                              <div><span className="text-gray-500">Numeric Columns:</span> <strong className="text-gray-900">{datasetPreview.numeric_columns?.length ?? 0}</strong></div>
                              <div><span className="text-gray-500">Categorical Columns:</span> <strong className="text-gray-900">{datasetPreview.categorical_columns?.length ?? 0}</strong></div>
                              <div><span className="text-gray-500">Duplicate Rows:</span> <strong className={datasetPreview.duplicate_rows > 0 ? 'text-amber-700' : 'text-green-700'}>{datasetPreview.duplicate_rows ?? 0}</strong></div>
                              <div>
                                <span className="text-gray-500">Missing Values:</span>{' '}
                                <strong className={Object.values(datasetPreview.missing_counts || {}).some((v: any) => v > 0) ? 'text-amber-700' : 'text-green-700'}>
                                  {Object.values(datasetPreview.missing_counts || {}).filter((v: any) => v > 0).length} cols affected
                                </strong>
                              </div>
                            </div>
                            <div className="pt-2 border-t border-slate-200">
                              <span className="text-gray-500 block mb-1">Audited Dataset Columns ({datasetPreview.columns?.length ?? 0}):</span>
                              <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                                {datasetPreview.columns?.map((c: string) => (
                                  <span key={c} className="bg-white border border-gray-200 px-1.5 py-0.5 rounded text-[10px] font-mono text-gray-800">
                                    {c}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <p className="text-gray-400">Select or upload a test dataset above to inspect full distribution properties.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="p-6 text-center text-xs text-gray-400 bg-slate-50 rounded-xl border border-dashed border-gray-200">
                      Model and dataset profiling will automatically appear here upon selection.
                    </div>
                  )}
                </div>

                {/* STEP 5 — SELECT VALIDATION MODULES & PRE-FLIGHT CHECK */}
                <div className="border-b border-gray-100 pb-6">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-semibold text-gray-800 flex items-center">
                      <span className="bg-blue-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center mr-2">5</span>
                      Select Validation Modules & Pre-Flight Compatibility
                    </label>
                    <span className="text-xs text-gray-500 font-medium">
                      {selectedValidators.length} of {VALIDATORS.length} modules enabled
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mb-4">Choose empirical pillars to evaluate. If prerequisites are not available, validators will transparently report &quot;Not Configured / Insufficient Information&quot; without fabricating results.</p>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
                    {VALIDATORS.map((val) => {
                      const isSelected = selectedValidators.includes(val.id);
                      return (
                        <div
                          key={val.id}
                          onClick={() => toggleValidator(val.id)}
                          className={`p-3.5 rounded-xl border text-left cursor-pointer transition-all flex flex-col justify-between ${
                            isSelected
                              ? 'border-blue-500 bg-blue-50/40 shadow-sm'
                              : 'border-gray-200 bg-white hover:border-gray-300 opacity-60'
                          }`}
                        >
                          <div>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-semibold text-gray-900">{val.name}</span>
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => {}}
                                className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                              />
                            </div>
                            <p className="text-xs text-gray-500 leading-relaxed mb-2">{val.description}</p>
                          </div>
                          <span className="text-[11px] font-medium text-blue-600">
                            Automatic evaluation
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Compatibility Pre-Flight Verification Box */}
                  {(() => {
                    const startButtonDisabled = startingRun || !selectedModel || !selectedDataset || (compatResult !== null && !compatResult.compatible);
                    const compatButtonDisabled = checkingCompat || !selectedModel || !selectedDataset;
                    console.log("VALIDATION DEBUG", {
                      projectId: selectedProject,
                      projectMode,
                      modelId: selectedModel,
                      datasetId: selectedDataset,
                      compatibilityStatus: compatResult?.compatible,
                      compatibilityPassed: compatResult ? compatResult.compatible : null,
                      selectedValidators,
                      checkingCompat,
                      startingRun,
                      startButtonDisabled,
                      compatButtonDisabled,
                    });
                    console.log("START BUTTON DISABLED:", startButtonDisabled);
                    console.log("COMPATIBILITY BUTTON DISABLED:", compatButtonDisabled);
                    return null;
                  })()}
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-sm font-bold text-gray-900 flex items-center">
                          Pre-Flight Compatibility Verification
                        </h4>
                        <p className="text-xs text-gray-500 mt-0.5">
                          Verifies deserialization, input feature alignment, NaN handling, and prediction interface across all samples.
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={handleCheckCompatibility}
                        disabled={checkingCompat || !selectedModel || !selectedDataset}
                        className="bg-white border border-gray-300 hover:bg-gray-50 text-gray-800 px-4 py-2 rounded-lg text-xs font-semibold shadow-sm flex items-center disabled:opacity-50"
                      >
                        <RefreshCw className={`h-3.5 w-3.5 mr-2 ${checkingCompat ? 'animate-spin' : ''}`} />
                        {checkingCompat ? 'Verifying Compatibility...' : 'Check Compatibility'}
                      </button>
                    </div>

                    {compatError && (
                      <div className="bg-red-50 border border-red-200 p-3 rounded-lg text-xs text-red-800 flex items-start">
                        <XCircle className="h-4 w-4 text-red-600 mr-2 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-semibold">Compatibility Check Failed</p>
                          <p className="mt-0.5">{compatError}</p>
                        </div>
                      </div>
                    )}

                    {compatResult && (
                      <div className={`p-4 rounded-xl border text-xs space-y-3 ${
                        compatResult.compatible ? 'bg-green-50/70 border-green-200' : 'bg-red-50/70 border-red-200'
                      }`}>
                        <div className="flex items-center justify-between">
                          <span className={`font-bold text-sm flex items-center ${
                            compatResult.compatible ? 'text-green-800' : 'text-red-800'
                          }`}>
                            {compatResult.compatible ? (
                              <>
                                <CheckCircle className="h-4 w-4 mr-1.5 text-green-600" />
                                ✓ Pre-Flight Checks Passed — Ready for Execution
                              </>
                            ) : (
                              <>
                                <XCircle className="h-4 w-4 mr-1.5 text-red-600" />
                                ❌ Incompatible Configuration — Resolution Required
                              </>
                            )}
                          </span>
                          <span className="text-[11px] text-gray-500">
                            Model: {compatResult.model_info?.class || 'Estimator'} • Features: {compatResult.dataset_info?.feature_count ?? 'N/A'}
                          </span>
                        </div>

                        {compatResult.errors?.length > 0 && (
                          <div className="space-y-1 bg-red-100/60 p-2.5 rounded border border-red-200 text-red-800">
                            <p className="font-semibold">Blocking Issues:</p>
                            {compatResult.errors.map((err, i) => (
                              <p key={i}>• {err}</p>
                            ))}
                          </div>
                        )}

                        {compatResult.warnings?.length > 0 && (
                          <div className="space-y-1 bg-amber-100/60 p-2.5 rounded border border-amber-200 text-amber-800">
                            <p className="font-semibold">Warnings / Automatic Advisories:</p>
                            {compatResult.warnings.map((warn, i) => (
                              <p key={i}>• {warn}</p>
                            ))}
                          </div>
                        )}

                        {/* Module Readiness Overview */}
                        <div className="pt-2 border-t border-slate-200/60 grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {compatResult.validator_readiness?.map(vr => (
                            <div key={vr.name} className="bg-white/80 p-2 rounded border border-gray-200/80">
                              <p className="font-semibold text-gray-900 capitalize">{vr.name.replace('_', ' ')}</p>
                              <p className={`text-[10px] font-medium mt-0.5 ${vr.ready ? 'text-green-700' : 'text-amber-700'}`}>
                                {vr.status_label}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* STEP 6 — START FULL VALIDATION */}
                {runError && (
                  <div className="bg-red-50 border border-red-200 p-4 rounded-xl flex items-start text-xs text-red-800">
                    <XCircle className="h-4 w-4 text-red-600 mr-2 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-semibold">Validation Could Not Start</p>
                      <p className="mt-0.5">{runError}</p>
                    </div>
                  </div>
                )}
                <div className="flex justify-between items-center bg-slate-900 text-white p-5 rounded-xl">
                  <div>
                    <h4 className="text-sm font-bold flex items-center">
                      <Play className="h-4 w-4 mr-2 text-blue-400 fill-current" /> Execute Full Model Assurance Suite
                    </h4>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Audits complete model predictions, data quality, robustness, explainability, calibration, and potential risks.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleStartValidation}
                    disabled={startingRun || !selectedModel || !selectedDataset || (compatResult !== null && !compatResult.compatible)}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2.5 rounded-lg text-sm font-semibold shadow flex items-center space-x-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Play className={`h-4 w-4 fill-current ${startingRun ? 'animate-spin' : ''}`} />
                    <span>{startingRun ? 'Executing Validation Suite...' : 'START FULL VALIDATION'}</span>
                  </button>
                </div>
              </>
            ) : (
              <div className="text-center py-6 text-sm text-gray-500">
                Please select or create a project context above to unlock model and dataset ingestion.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Existing Validation Runs History Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h3 className="text-base font-semibold text-gray-900">Recent Validation Runs</h3>
          <span className="text-xs text-gray-500">{runs.length} runs recorded</span>
        </div>
        {loading ? (
          <div className="p-12 text-center text-gray-500">Loading validation history...</div>
        ) : (
          <DataTable
            columns={[
              {
                key: 'run_number',
                header: 'Run Number',
                render: (r) => (
                  <span className="font-mono text-blue-600 font-semibold">{r.run_number}</span>
                )
              },
              {
                key: 'status',
                header: 'Status',
                render: (r) => <StatusBadge status={r.status} />
              },
              {
                key: 'overall_score',
                header: 'Overall Score',
                render: (r) => (
                  <span className="font-semibold text-gray-900">
                    {r.overall_score !== null && r.overall_score !== undefined
                      ? `${Math.round(r.overall_score)}/100`
                      : '-'}
                  </span>
                )
              },
              {
                key: 'started_at',
                header: 'Started At',
                render: (r) => r.started_at ? new Date(r.started_at).toLocaleString() : '-'
              },
              {
                key: 'duration_seconds',
                header: 'Duration',
                render: (r) => r.duration_seconds ? `${r.duration_seconds.toFixed(2)}s` : '-'
              },
              {
                key: 'actions',
                header: 'Actions',
                render: (r) => (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/validation-runs/${r.id}`);
                    }}
                    className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center"
                  >
                    <Eye className="h-3.5 w-3.5 mr-1" />
                    View Results
                  </button>
                )
              }
            ]}
            data={runs}
            onRowClick={(run) => navigate(`/validation-runs/${run.id}`)}
            emptyMessage="No validation runs recorded yet. Start your first run above."
          />
        )}
      </div>
    </div>
  );
}
