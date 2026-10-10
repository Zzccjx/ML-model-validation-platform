import React, { useState, useEffect } from 'react';
import api from '../services/api';
import DataTable from '../components/DataTable';
import FileUpload from '../components/FileUpload';
import { MLModel, Project } from '../types';

export default function ModelsPage() {
  const [models, setModels] = useState<MLModel[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [selectedProject, setSelectedProject] = useState('');
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchModels();
    fetchProjects();
  }, []);

  const fetchModels = async () => {
    try {
      const { data } = await api.get<MLModel[]>('/models');
      setModels(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchProjects = async () => {
    try {
      const { data } = await api.get<Project[]>('/projects');
      setProjects(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!selectedProject) {
      alert('Please select a project first');
      return;
    }
    
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name.replace(/\.[^/.]+$/, ''));
    formData.append('project_id', selectedProject);

    try {
      await api.post('/models/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setShowUpload(false);
      fetchModels();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Models</h1>
          <p className="text-gray-500">Manage uploaded machine learning models</p>
        </div>
        <button
          onClick={() => setShowUpload(!showUpload)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700"
        >
          {showUpload ? 'Cancel' : 'Upload Model'}
        </button>
      </div>

      {showUpload && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
          <h3 className="text-lg font-medium mb-4">Upload New Model</h3>
          <div className="max-w-xl space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Project</label>
              <select
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md border"
              >
                <option value="">Select a project...</option>
                {projects.map(p => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </div>
            
            {selectedProject && (
              <div className="mt-4">
                {uploading ? (
                  <div className="text-center py-8">Uploading...</div>
                ) : (
                  <FileUpload 
                    accept=".pkl,.joblib,.pickle,.keras,.h5,.onnx"
                    onFileSelect={handleFileUpload}
                    label="Drag and drop model file here"
                    description="Supports .pkl, .joblib, .pickle, .keras, .h5, .onnx"
                  />
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {loading ? (
          <div className="p-12 text-center text-gray-500">Loading...</div>
        ) : (
          <DataTable
            columns={[
              { key: 'name', header: 'Name' },
              { key: 'model_type', header: 'Type' },
              { key: 'framework', header: 'Framework' },
              { key: 'feature_count', header: 'Features', render: (m) => m.feature_count || '-' },
              { key: 'uploaded_at', header: 'Uploaded', render: (m) => new Date(m.uploaded_at).toLocaleDateString() }
            ]}
            data={models}
            emptyMessage="No models found. Upload one to get started."
          />
        )}
      </div>
    </div>
  );
}
