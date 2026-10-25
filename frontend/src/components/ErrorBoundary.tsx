import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an unhandled render error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-[400px] flex items-center justify-center p-6 bg-slate-50">
          <div className="max-w-xl w-full bg-white rounded-2xl shadow-lg border border-red-100 p-6 sm:p-8 space-y-5">
            <div className="flex items-center space-x-3 text-red-600">
              <div className="w-12 h-12 rounded-xl bg-red-50 flex items-center justify-center border border-red-200">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-gray-900">
                  {this.props.fallbackTitle || 'Render Error Prevented'}
                </h2>
                <p className="text-xs text-gray-500">
                  An unexpected render exception was caught safely by the system.
                </p>
              </div>
            </div>

            {this.state.error && (
              <div className="bg-red-50/70 border border-red-200 rounded-xl p-4 text-xs font-mono text-red-900 overflow-x-auto max-h-48">
                <div className="font-bold text-red-800 mb-1">
                  {this.state.error.name}: {this.state.error.message}
                </div>
                {this.state.error.stack && (
                  <pre className="text-[11px] text-red-700 whitespace-pre-wrap leading-relaxed mt-2 opacity-80">
                    {this.state.error.stack.split('\n').slice(0, 5).join('\n')}
                  </pre>
                )}
              </div>
            )}

            <div className="flex items-center justify-between pt-2 border-t border-gray-100">
              <a
                href="/"
                className="inline-flex items-center text-xs font-semibold text-gray-600 hover:text-gray-900 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <Home className="w-4 h-4 mr-1.5" />
                Return to Dashboard
              </a>

              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={this.handleReset}
                  className="inline-flex items-center text-xs font-semibold text-gray-700 bg-gray-100 hover:bg-gray-200 px-3 py-2 rounded-lg transition-colors"
                >
                  Try Again
                </button>
                <button
                  type="button"
                  onClick={this.handleReload}
                  className="inline-flex items-center text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg shadow-sm transition-colors"
                >
                  <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
                  Reload Page
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
