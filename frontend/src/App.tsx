import React, { useState, useEffect, useRef } from 'react';
import { Search, Settings, Download, AlertCircle, CheckCircle2, Loader2, Brain, FileText, Clock, Zap } from 'lucide-react';
import './App.css';

// Types
interface Model {
  name: string;
  provider: string;
  description?: string;
  api_key_available?: boolean;
}

interface Models {
  [provider: string]: {
    [key: string]: Model;
  };
}

interface ResearchRequest {
  query: string;
  provider: string;
  model_key: string;
  max_tokens?: number;
  temperature?: number;
  output_filename?: string;
}

interface ProgressUpdate {
  type: string;
  session_id?: string;
  stage?: string;
  progress?: number;
  elapsed_time?: number;
  estimated_remaining?: number;
  message?: string;
  result?: any;
}

const App: React.FC = () => {
  // State management
  const [query, setQuery] = useState('');
  const [models, setModels] = useState<Models>({});
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [maxTokens, setMaxTokens] = useState(500);
  const [temperature, setTemperature] = useState(0.2);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [estimatedRemaining, setEstimatedRemaining] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // API Base URL
  const API_BASE = 'http://localhost:8000';
  const WS_BASE = 'ws://localhost:8000';

  // Load available models on component mount
  useEffect(() => {
    loadModels();
  }, []);

  // Timer for elapsed time
  useEffect(() => {
    if (isLoading) {
      timerRef.current = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setElapsedTime(0);
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isLoading]);

  const loadModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/models`);
      const data = await response.json();
      
      if (data.success) {
        setModels(data.models);
        
        // Auto-select first available model
        const providers = Object.keys(data.models);
        if (providers.length > 0) {
          const firstProvider = providers[0];
          setSelectedProvider(firstProvider);
          
          const modelKeys = Object.keys(data.models[firstProvider]);
          if (modelKeys.length > 0) {
            setSelectedModel(modelKeys[0]);
          }
        }
      }
    } catch (err) {
      setError('Failed to load models. Make sure the backend is running.');
      console.error('Error loading models:', err);
    }
  };

  const startResearch = async () => {
    if (!query.trim() || !selectedProvider || !selectedModel) {
      setError('Please fill in all required fields');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);
    setProgress(0);
    setCurrentStage('');

    try {
      // Create research session
      const request: ResearchRequest = {
        query: query.trim(),
        provider: selectedProvider,
        model_key: selectedModel,
        max_tokens: maxTokens,
        temperature: temperature
      };

      const response = await fetch(`${API_BASE}/api/research/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to start research');
      }

      const newSessionId = data.session_id;
      setSessionId(newSessionId);

      // Connect to WebSocket for progress updates
      connectWebSocket(newSessionId);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setIsLoading(false);
    }
  };

  const connectWebSocket = (sessionId: string) => {
    const ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data: ProgressUpdate = JSON.parse(event.data);
      
      switch (data.type) {
        case 'progress':
          setProgress(data.progress || 0);
          setCurrentStage(data.stage || '');
          if (data.estimated_remaining) {
            setEstimatedRemaining(data.estimated_remaining);
          }
          break;
          
        case 'complete':
          setResults(data.result);
          setProgress(100);
          setCurrentStage('Completed!');
          setIsLoading(false);
          break;
          
        case 'error':
          setError(data.message || 'An error occurred');
          setIsLoading(false);
          break;
          
        case 'status':
          setCurrentStage(data.message || '');
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error occurred');
      setIsLoading(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      wsRef.current = null;
    };
  };

  const downloadReport = async () => {
    if (!results?.report_path) return;
    
    try {
      const filename = results.report_path.split('/').pop();
      const response = await fetch(`${API_BASE}/api/reports/${filename}`);
      
      if (!response.ok) {
        throw new Error('Failed to download report');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to download report');
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getProviderName = (provider: string): string => {
    const names: { [key: string]: string } = {
      'local': 'Local Models',
      'openai': 'OpenAI',
      'anthropic': 'Anthropic',
      'google': 'Google',
      'groq': 'Groq'
    };
    return names[provider] || provider;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-4 rounded-full mr-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white">AI Research Agent</h1>
          </div>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Generate comprehensive research reports using advanced AI models. 
            Simply enter your research question and let AI do the heavy lifting.
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          {/* Query Input */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-8 border border-white/20">
            <div className="mb-6">
              <label className="block text-white text-lg font-semibold mb-3">
                Research Question
              </label>
              <div className="relative">
                <Search className="absolute left-4 top-4 w-5 h-5 text-gray-400" />
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="What would you like to research? E.g., 'Software engineer salaries in Portugal 2024' or 'Compare React vs Vue.js for enterprise development'"
                  className="w-full pl-12 pr-4 py-4 bg-black/20 border border-white/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={3}
                  disabled={isLoading}
                />
              </div>
            </div>

            {/* Model Selection */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div>
                <label className="block text-white text-sm font-medium mb-2">
                  AI Provider
                </label>
                <select
                  value={selectedProvider}
                  onChange={(e) => {
                    setSelectedProvider(e.target.value);
                    const firstModel = Object.keys(models[e.target.value] || {})[0];
                    setSelectedModel(firstModel || '');
                  }}
                  className="w-full px-4 py-3 bg-black/20 border border-white/30 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading}
                >
                  <option value="">Select Provider</option>
                  {Object.keys(models).map(provider => (
                    <option key={provider} value={provider}>
                      {getProviderName(provider)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-white text-sm font-medium mb-2">
                  Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-4 py-3 bg-black/20 border border-white/30 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isLoading || !selectedProvider}
                >
                  <option value="">Select Model</option>
                  {selectedProvider && models[selectedProvider] && 
                    Object.entries(models[selectedProvider]).map(([key, model]) => (
                      <option key={key} value={key}>
                        {model.name}
                      </option>
                    ))
                  }
                </select>
              </div>
            </div>

            {/* Advanced Settings */}
            <div className="mb-6">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center text-blue-400 hover:text-blue-300 transition-colors"
                disabled={isLoading}
              >
                <Settings className="w-4 h-4 mr-2" />
                Advanced Settings
              </button>

              {showAdvanced && (
                <div className="mt-4 grid md:grid-cols-2 gap-4 p-4 bg-black/20 rounded-xl border border-white/20">
                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Max Tokens: {maxTokens}
                    </label>
                    <input
                      type="range"
                      min="100"
                      max="2000"
                      step="100"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                      className="w-full"
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <label className="block text-white text-sm font-medium mb-2">
                      Temperature: {temperature}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                      className="w-full"
                      disabled={isLoading}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Start Research Button */}
            <button
              onClick={startResearch}
              disabled={isLoading || !query.trim() || !selectedProvider || !selectedModel}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-200 flex items-center justify-center space-x-2 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Researching...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Start Research</span>
                </>
              )}
            </button>
          </div>

          {/* Progress Section */}
          {isLoading && (
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-8 border border-white/20">
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-medium">{currentStage || 'Processing...'}</span>
                  <span className="text-blue-400 font-mono">{progress}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="flex items-center justify-between text-sm text-gray-300">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    <span>Elapsed: {formatTime(elapsedTime)}</span>
                  </div>
                  {estimatedRemaining && (
                    <div className="flex items-center">
                      <span>ETA: {formatTime(Math.round(estimatedRemaining))}</span>
                    </div>
                  )}
                </div>
                <div className="text-blue-400">Session: {sessionId?.slice(-8)}</div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/20 border border-red-500/50 rounded-2xl p-6 mb-8">
              <div className="flex items-center">
                <AlertCircle className="w-6 h-6 text-red-400 mr-3" />
                <div>
                  <h3 className="text-red-400 font-semibold mb-1">Error</h3>
                  <p className="text-red-300">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          {results && (
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-8 border border-white/20">
              <div className="flex items-center mb-6">
                <CheckCircle2 className="w-6 h-6 text-green-400 mr-3" />
                <h2 className="text-2xl font-bold text-white">Research Complete!</h2>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-black/20 rounded-xl p-4 border border-white/20">
                  <div className="text-2xl font-bold text-blue-400 mb-1">
                    {results.content_length?.toLocaleString() || 'N/A'}
                  </div>
                  <div className="text-gray-300 text-sm">Characters Generated</div>
                </div>

                <div className="bg-black/20 rounded-xl p-4 border border-white/20">
                  <div className="text-2xl font-bold text-purple-400 mb-1">
                    {results.categories?.length || 0}
                  </div>
                  <div className="text-gray-300 text-sm">Research Categories</div>
                </div>

                <div className="bg-black/20 rounded-xl p-4 border border-white/20">
                  <div className="text-2xl font-bold text-green-400 mb-1">
                    {results.report_structure?.length || 0}
                  </div>
                  <div className="text-gray-300 text-sm">Report Sections</div>
                </div>

                <div className="bg-black/20 rounded-xl p-4 border border-white/20">
                  <div className="text-2xl font-bold text-yellow-400 mb-1">
                    {results.pdf_created ? 'Yes' : 'No'}
                  </div>
                  <div className="text-gray-300 text-sm">PDF Generated</div>
                </div>
              </div>

              {results.categories && results.categories.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-white font-semibold mb-3">Research Categories</h3>
                  <div className="flex flex-wrap gap-2">
                    {results.categories.map((category: string, index: number) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-sm border border-blue-500/30"
                      >
                        {category}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {results.report_path && (
                <button
                  onClick={downloadReport}
                  className="w-full md:w-auto bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white font-semibold py-3 px-8 rounded-xl transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  <Download className="w-5 h-5" />
                  <span>Download PDF Report</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;