import React, { useState, useEffect, useRef } from 'react';
import { Search, Settings, Download, AlertCircle, CheckCircle2, Loader2, Brain, FileText, Clock, Zap, Star, Code, BookOpen, Cpu, CheckCircle, HardDrive } from 'lucide-react';
import './App.css';

// Add NodeJS namespace
declare global {
  namespace NodeJS {
    type Timeout = number;
  }
}

// Types
interface Model {
  name: string;
  provider: string;
  description?: string;
  api_key_available?: boolean;
  best_for?: string;
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

// Enhanced Model Selection Component
const EnhancedModelSelector = ({ 
  models, 
  selectedProvider, 
  selectedModel, 
  setSelectedProvider, 
  setSelectedModel,
  query,
  isLoading 
}: {
  models: Models;
  selectedProvider: string;
  selectedModel: string;
  setSelectedProvider: (provider: string) => void;
  setSelectedModel: (model: string) => void;
  query: string;
  isLoading: boolean;
}) => {
  const [autoSelectEnabled, setAutoSelectEnabled] = useState(true);
  const [recommendedModel, setRecommendedModel] = useState('');

  // Auto-select best model based on query
  useEffect(() => {
    if (autoSelectEnabled && query && models.groq) {
      const recommended = autoSelectModel(query);
      setRecommendedModel(recommended);
      
      if (selectedProvider === 'groq' && recommended !== selectedModel) {
        setSelectedModel(recommended);
      }
    }
  }, [query, autoSelectEnabled, models, selectedProvider]);

  const autoSelectModel = (query: string) => {
    const queryWords = query.split(' ').length;
    const queryLower = query.toLowerCase();
    
    // Technical queries
    if (queryLower.includes('code') || queryLower.includes('programming') || queryLower.includes('software')) {
      return 'mixtral-8x7b-32768';
    }
    // Academic/research queries
    else if (queryLower.includes('research') || queryLower.includes('analysis') || queryLower.includes('study')) {
      return 'llama-3.2-90b-text-preview';
    }
    // Long complex queries
    else if (queryWords > 30) {
      return 'llama-3.1-70b-versatile';
    }
    // Medium queries
    else if (queryWords > 10) {
      return 'llama-3.1-70b-versatile';
    }
    // Quick queries
    else {
      return 'llama-3.1-8b-instant';
    }
  };

  const getModelIcon = (modelKey: string) => {
    if (modelKey.includes('70b') || modelKey.includes('90b')) return <Star className="w-4 h-4 text-yellow-400" />;
    if (modelKey.includes('mixtral')) return <Code className="w-4 h-4 text-purple-400" />;
    if (modelKey.includes('instant')) return <Zap className="w-4 h-4 text-blue-400" />;
    if (modelKey.includes('gemma')) return <Cpu className="w-4 h-4 text-green-400" />;
    return <Brain className="w-4 h-4 text-gray-400" />;
  };

  const getQualityBadge = (modelName: string) => {
    if (modelName.includes('⭐')) return 'premium';
    if (modelName.includes('🚀') || modelName.includes('⚡') || modelName.includes('💻')) return 'standard';
    return 'basic';
  };

  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'premium': return 'border-yellow-500/50 bg-yellow-500/10 text-yellow-300';
      case 'standard': return 'border-blue-500/50 bg-blue-500/10 text-blue-300';
      default: return 'border-gray-500/50 bg-gray-500/10 text-gray-300';
    }
  };

  // Check if provider is local (doesn't require API key)
  const isLocalProvider = (provider: string) => {
    return provider === 'local' || provider === 'ollama' || provider === 'lmstudio';
  };

  // Get API key status for the selected provider
  const getApiKeyStatus = () => {
    if (!selectedProvider) return null;
    
    if (isLocalProvider(selectedProvider)) {
      return {
        available: true,
        isLocal: true,
        message: 'Local Model - No API Key Required'
      };
    }
    
    // For external providers, check if API key is available
    const firstModel = models[selectedProvider] && Object.values(models[selectedProvider])[0];
    return {
      available: firstModel?.api_key_available || false,
      isLocal: false,
      message: firstModel?.api_key_available ? 'API Key Configured' : 'API Key Required'
    };
  };

  const apiKeyStatus = getApiKeyStatus();

  return (
    <div className="space-y-6">
      {/* Auto-Select Toggle */}
      <div className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/20">
        <div className="flex items-center space-x-3">
          <Brain className="w-5 h-5 text-blue-400" />
          <div>
            <h3 className="text-white font-medium">Smart Model Selection</h3>
            <p className="text-gray-400 text-sm">Automatically choose the best model for your query</p>
          </div>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={autoSelectEnabled}
            onChange={(e) => setAutoSelectEnabled(e.target.checked)}
            className="sr-only"
            disabled={isLoading}
          />
          <div className={`w-11 h-6 rounded-full ${autoSelectEnabled ? 'bg-blue-600' : 'bg-gray-600'} relative transition-colors`}>
            <div className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-transform ${autoSelectEnabled ? 'translate-x-6' : 'translate-x-1'}`}></div>
          </div>
        </label>
      </div>

      {/* Recommended Model Alert */}
      {autoSelectEnabled && recommendedModel && query && (
        <div className="flex items-center p-4 bg-green-500/10 border border-green-500/30 rounded-xl">
          <CheckCircle className="w-5 h-5 text-green-400 mr-3" />
          <div>
            <p className="text-green-300 font-medium">Recommended Model</p>
            <p className="text-green-200 text-sm">
              {models.groq?.[recommendedModel]?.name || recommendedModel} - Best fit for your query
            </p>
          </div>
        </div>
      )}

      {/* Provider Selection */}
      <div>
        <label className="block text-white text-sm font-medium mb-3">
          AI Provider
        </label>
        <select
          value={selectedProvider}
          onChange={(e) => {
            setSelectedProvider(e.target.value);
            if (e.target.value === 'groq' && autoSelectEnabled && recommendedModel) {
              setSelectedModel(recommendedModel);
            } else {
              const firstModel = Object.keys(models[e.target.value] || {})[0];
              setSelectedModel(firstModel || '');
            }
          }}
          className="w-full px-4 py-3 bg-black/20 border border-white/30 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        >
          <option value="">Select Provider</option>
          {Object.keys(models).map(provider => (
            <option key={provider} value={provider}>
              {provider === 'groq' ? '🚀 Groq (Recommended)' : 
               provider === 'local' ? '💻 Local Models' :
               provider === 'ollama' ? '🏠 Ollama' :
               provider === 'lmstudio' ? '🖥️ LM Studio' : 
               provider}
            </option>
          ))}
        </select>
      </div>

      {/* Enhanced Model Selection */}
      {selectedProvider && (
        <div>
          <label className="block text-white text-sm font-medium mb-3">
            Model Selection
          </label>
          <div className="space-y-3">
            {selectedProvider && models[selectedProvider] && 
              Object.entries(models[selectedProvider]).map(([key, model]) => {
                const quality = getQualityBadge(model.name);
                const isSelected = selectedModel === key;
                const isRecommended = key === recommendedModel;
                
                return (
                  <div
                    key={key}
                    className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                      isSelected 
                        ? 'border-blue-500 bg-blue-500/10' 
                        : isRecommended
                        ? 'border-green-500/50 bg-green-500/5'
                        : 'border-white/20 bg-black/10 hover:border-white/40'
                    }`}
                    onClick={() => !isLoading && setSelectedModel(key)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          {getModelIcon(key)}
                          <h3 className="text-white font-medium">{model.name}</h3>
                          {isRecommended && (
                            <span className="px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded-full border border-green-500/30">
                              Recommended
                            </span>
                          )}
                          <span className={`px-2 py-1 text-xs rounded-full border ${getQualityColor(quality)}`}>
                            {quality}
                          </span>
                        </div>
                        <p className="text-gray-300 text-sm mb-2">{model.description}</p>
                        {model.best_for && (
                          <p className="text-gray-400 text-xs">
                            <strong>Best for:</strong> {model.best_for}
                          </p>
                        )}
                      </div>
                      <div className="ml-4">
                        <input
                          type="radio"
                          checked={isSelected}
                          onChange={() => setSelectedModel(key)}
                          className="w-4 h-4 text-blue-600"
                          disabled={isLoading}
                        />
                      </div>
                    </div>
                  </div>
                );
              })
            }
          </div>
        </div>
      )}

      {/* Model Comparison */}
      {selectedProvider === 'groq' && (
        <div className="bg-black/20 rounded-xl p-4 border border-white/20">
          <h3 className="text-white font-medium mb-3 flex items-center">
            <Brain className="w-4 h-4 mr-2" />
            Model Comparison
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-yellow-400 font-medium mb-1">Premium Models</div>
              <div className="text-gray-300">
                • Best quality<br/>
                • Complex analysis<br/>
                • Detailed reports
              </div>
            </div>
            <div className="text-center">
              <div className="text-blue-400 font-medium mb-1">Standard Models</div>
              <div className="text-gray-300">
                • Good balance<br/>
                • Fast processing<br/>
                • General research
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400 font-medium mb-1">Basic Models</div>
              <div className="text-gray-300">
                • Ultra fast<br/>
                • Simple queries<br/>
                • Quick testing
              </div>
            </div>
          </div>
        </div>
      )}

      {/* API Key Status */}
      {apiKeyStatus && (
        <div className="flex items-center p-3 bg-black/20 rounded-xl border border-white/20">
          {apiKeyStatus.isLocal ? (
            <>
              <HardDrive className="w-4 h-4 text-blue-400 mr-2" />
              <span className="text-blue-300 text-sm">{apiKeyStatus.message}</span>
            </>
          ) : apiKeyStatus.available ? (
            <>
              <CheckCircle className="w-4 h-4 text-green-400 mr-2" />
              <span className="text-green-300 text-sm">{apiKeyStatus.message}</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-4 h-4 text-yellow-400 mr-2" />
              <span className="text-yellow-300 text-sm">{apiKeyStatus.message}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
};

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
      'groq': 'Groq',
      'ollama': 'Ollama',
      'lmstudio': 'LM Studio'
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
            <EnhancedModelSelector
              models={models}
              selectedProvider={selectedProvider}
              selectedModel={selectedModel}
              setSelectedProvider={setSelectedProvider}
              setSelectedModel={setSelectedModel}
              query={query}
              isLoading={isLoading}
            />

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