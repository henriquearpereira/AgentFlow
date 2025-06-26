import React, { useState, useEffect, useRef } from 'react';
import { Search, Settings, Download, AlertCircle, CheckCircle2, Loader2, Brain, FileText, Clock, Zap, Star, Code, BookOpen, Cpu, CheckCircle, HardDrive, Search as SearchIcon, BarChart3 } from 'lucide-react';
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
  best_for?: string[];
  estimated_time?: string;
  cost?: string;
  quality?: string;
  max_tokens?: number;
  recommended_temperature?: number;
  quality_tier?: string;
  memory_usage?: string;
  id?: string;
  object?: string;
  created?: number;
  owned_by?: string;
  active?: boolean;
  context_window?: number;
  public_apps?: any;
  max_completion_tokens?: number;
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
  search_engine: string;
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
  const [filterQuality, setFilterQuality] = useState<string>('all');
  const [filterContext, setFilterContext] = useState<string>('all');

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

  // Filter models based on selected criteria
  const getFilteredModels = () => {
    if (!selectedProvider || !models[selectedProvider]) return [];
    
    let filtered = Object.entries(models[selectedProvider]);
    
    // Filter by quality tier
    if (filterQuality !== 'all') {
      filtered = filtered.filter(([key, model]) => model.quality_tier === filterQuality);
    }
    
    // Filter by context window
    if (filterContext !== 'all') {
      const contextThresholds = {
        'ultra': 131072,
        'high': 32768,
        'medium': 8192,
        'basic': 0
      };
      const threshold = contextThresholds[filterContext as keyof typeof contextThresholds];
      filtered = filtered.filter(([key, model]) => 
        model.context_window && model.context_window >= threshold
      );
    }
    
    return filtered;
  };

  const autoSelectModel = (query: string) => {
    const queryWords = query.split(' ').length;
    const queryLower = query.toLowerCase();
    
    // Get available models for the selected provider
    const availableModels = models[selectedProvider] || {};
    const modelEntries = Object.entries(availableModels);
    
    if (modelEntries.length === 0) return '';
    
    // Technical queries - prefer models with larger context windows
    if (queryLower.includes('code') || queryLower.includes('programming') || queryLower.includes('software') || queryLower.includes('api') || queryLower.includes('technical')) {
      // Find model with largest context window for technical tasks
      const technicalModels = modelEntries.filter(([key, model]) => 
        model.context_window && model.context_window >= 8192
      );
      if (technicalModels.length > 0) {
        return technicalModels.reduce((best, current) => 
          (current[1].context_window || 0) > (best[1].context_window || 0) ? current : best
        )[0];
      }
      return 'compound-beta';
    }
    
    // Academic/research queries - prefer high-quality models
    else if (queryLower.includes('research') || queryLower.includes('analysis') || queryLower.includes('study') || queryLower.includes('academic') || queryLower.includes('thesis')) {
      // Prefer models with "premium" quality tier or large context windows
      const academicModels = modelEntries.filter(([key, model]) => 
        model.quality_tier === 'premium' || (model.context_window && model.context_window >= 32768)
      );
      if (academicModels.length > 0) {
        return academicModels[0][0];
      }
      return 'llama-3.3-70b-versatile';
    }
    
    // Long complex queries - need large context windows
    else if (queryWords > 30) {
      const largeContextModels = modelEntries.filter(([key, model]) => 
        model.context_window && model.context_window >= 32768
      );
      if (largeContextModels.length > 0) {
        return largeContextModels[0][0];
      }
      return 'llama-3.3-70b-versatile';
    }
    
    // Medium queries
    else if (queryWords > 10) {
      const mediumModels = modelEntries.filter(([key, model]) => 
        model.context_window && model.context_window >= 8192
      );
      if (mediumModels.length > 0) {
        return mediumModels[0][0];
      }
      return 'llama-3.3-70b-versatile';
    }
    
    // Quick queries - prefer fast models
    else {
      const fastModels = modelEntries.filter(([key, model]) => 
        key.includes('instant') || key.includes('3b') || key.includes('8b')
      );
      if (fastModels.length > 0) {
        return fastModels[0][0];
      }
      return 'llama-3.1-8b-instant';
    }
  };

  const getModelIcon = (modelKey: string) => {
    if (modelKey.includes('70b') || modelKey.includes('90b')) return <Star className="w-4 h-4 text-yellow-400" />;
    if (modelKey.includes('mixtral')) return <Code className="w-4 h-4 text-purple-400" />;
    if (modelKey.includes('instant') || modelKey.includes('3b')) return <Zap className="w-4 h-4 text-blue-400" />;
    if (modelKey.includes('gemma')) return <Cpu className="w-4 h-4 text-green-400" />;
    if (modelKey.includes('whisper')) return <BookOpen className="w-4 h-4 text-orange-400" />;
    if (modelKey.includes('compound')) return <Brain className="w-4 h-4 text-indigo-400" />;
    if (modelKey.includes('qwen')) return <Cpu className="w-4 h-4 text-teal-400" />;
    if (modelKey.includes('llama')) return <Brain className="w-4 h-4 text-gray-400" />;
    return <Brain className="w-4 h-4 text-gray-400" />;
  };

  const getQualityBadge = (qualityTier: string | undefined) => {
    switch (qualityTier) {
      case 'premium': return 'Premium';
      case 'standard': return 'Standard';
      case 'basic': return 'Basic';
      default: return 'Unknown';
    }
  };

  const getQualityColor = (quality: string | undefined) => {
    switch (quality) {
      case 'premium': return 'border-yellow-500/50 bg-yellow-500/10 text-yellow-300';
      case 'standard': return 'border-blue-500/50 bg-blue-500/10 text-blue-300';
      case 'basic': return 'border-gray-500/50 bg-gray-500/10 text-gray-300';
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

  // Model Capabilities Component
  const ModelCapabilities = ({ model }: { model: Model }) => {
    const getCapabilityLevel = (contextWindow: number | undefined) => {
      if (!contextWindow) return 'Unknown';
      if (contextWindow >= 131072) return 'Ultra';
      if (contextWindow >= 32768) return 'High';
      if (contextWindow >= 8192) return 'Medium';
      return 'Basic';
    };

    const getCapabilityColor = (level: string) => {
      switch (level) {
        case 'Ultra': return 'text-purple-400';
        case 'High': return 'text-yellow-400';
        case 'Medium': return 'text-blue-400';
        case 'Basic': return 'text-gray-400';
        default: return 'text-gray-400';
      }
    };

    return (
      <div className="mt-3 p-3 bg-black/10 rounded-lg border border-white/10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          {model.context_window && (
            <div>
              <div className="text-gray-400 mb-1">Context Window</div>
              <div className={`font-mono ${getCapabilityColor(getCapabilityLevel(model.context_window))}`}>
                {model.context_window.toLocaleString()}
              </div>
            </div>
          )}
          {model.max_completion_tokens && (
            <div>
              <div className="text-gray-400 mb-1">Max Completion</div>
              <div className="text-blue-300 font-mono">
                {model.max_completion_tokens.toLocaleString()}
              </div>
            </div>
          )}
          {model.owned_by && (
            <div>
              <div className="text-gray-400 mb-1">Provider</div>
              <div className="text-green-300">{model.owned_by}</div>
            </div>
          )}
          {model.quality_tier && (
            <div>
              <div className="text-gray-400 mb-1">Quality</div>
              <div className={`capitalize ${getQualityColor(model.quality_tier).split(' ')[2]}`}>
                {model.quality_tier}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

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
            {models.groq?.[recommendedModel]?.context_window && (
              <p className="text-green-200 text-xs mt-1">
                Context window: {models.groq[recommendedModel].context_window?.toLocaleString()} tokens
              </p>
            )}
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
          
          {/* Model Filters */}
          <div className="mb-4 flex flex-wrap gap-4">
            <div>
              <label className="block text-gray-300 text-xs mb-1">Quality Tier</label>
              <select
                value={filterQuality}
                onChange={(e) => setFilterQuality(e.target.value)}
                className="px-3 py-1 bg-black/20 border border-white/30 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                disabled={isLoading}
              >
                <option value="all">All Tiers</option>
                <option value="premium">Premium</option>
                <option value="standard">Standard</option>
                <option value="basic">Basic</option>
              </select>
            </div>
            
            <div>
              <label className="block text-gray-300 text-xs mb-1">Context Window</label>
              <select
                value={filterContext}
                onChange={(e) => setFilterContext(e.target.value)}
                className="px-3 py-1 bg-black/20 border border-white/30 rounded-lg text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                disabled={isLoading}
              >
                <option value="all">All Sizes</option>
                <option value="ultra">Ultra (131K+)</option>
                <option value="high">High (32K+)</option>
                <option value="medium">Medium (8K+)</option>
                <option value="basic">Basic (Any)</option>
              </select>
            </div>
          </div>
          
          <div className="space-y-3">
            {selectedProvider && models[selectedProvider] && 
              getFilteredModels().map(([key, model]) => {
                const quality = getQualityBadge(model.quality_tier);
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
                          <span className={`px-2 py-1 text-xs rounded-full border ${getQualityColor(model.quality_tier)}`}>
                            {quality}
                          </span>
                          {model.active !== undefined && (
                            <span className={`px-2 py-1 text-xs rounded-full border ${
                              model.active 
                                ? 'border-green-500/50 bg-green-500/10 text-green-300' 
                                : 'border-red-500/50 bg-red-500/10 text-red-300'
                            }`}>
                              {model.active ? 'Active' : 'Inactive'}
                            </span>
                          )}
                        </div>
                        <p className="text-gray-300 text-sm mb-2">{model.description}</p>
                        {/* Show best_for as tags */}
                        {model.best_for && model.best_for.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-2">
                            {model.best_for.map((tag, idx) => (
                              <span key={idx} className="px-2 py-0.5 bg-blue-500/10 text-blue-300 rounded-full text-xs border border-blue-500/20">
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                        {/* Show estimated_time, cost, max_tokens, recommended_temperature, memory_usage */}
                        <div className="flex flex-wrap gap-4 text-xs text-gray-400">
                          {model.estimated_time && <span>⏱️ {model.estimated_time}</span>}
                          {model.cost && <span>💲{model.cost}</span>}
                          {model.max_tokens && <span>🔢 Max tokens: {model.max_tokens}</span>}
                          {model.recommended_temperature !== undefined && <span>🌡️ Temp: {model.recommended_temperature}</span>}
                          {model.memory_usage && <span>💾 {model.memory_usage}</span>}
                          {model.context_window && <span>📄 Context: {model.context_window.toLocaleString()}</span>}
                          {model.max_completion_tokens && <span>✍️ Completion: {model.max_completion_tokens.toLocaleString()}</span>}
                          {model.owned_by && <span>🏢 {model.owned_by}</span>}
                          {model.created && (
                            <span>📅 {new Date(model.created * 1000).toLocaleDateString()}</span>
                          )}
                        </div>
                        
                        {/* Model Capabilities */}
                        {(model.context_window || model.max_completion_tokens || model.owned_by) && (
                          <ModelCapabilities model={model} />
                        )}
                        
                        {/* Model Performance Metrics */}
                        {(model.estimated_time || model.quality_tier || model.context_window) && (
                          <div className="mt-3 p-3 bg-black/10 rounded-lg border border-white/10">
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                              {model.estimated_time && (
                                <div>
                                  <div className="text-gray-400 mb-1">Speed</div>
                                  <div className="text-yellow-300 font-medium">{model.estimated_time}</div>
                                </div>
                              )}
                              {model.quality_tier && (
                                <div>
                                  <div className="text-gray-400 mb-1">Quality</div>
                                  <div className={`capitalize font-medium ${getQualityColor(model.quality_tier).split(' ')[2]}`}>
                                    {model.quality_tier}
                                  </div>
                                </div>
                              )}
                              {model.context_window && (
                                <div>
                                  <div className="text-gray-400 mb-1">Capacity</div>
                                  <div className="text-blue-300 font-medium">
                                    {model.context_window >= 131072 ? 'Ultra' :
                                     model.context_window >= 32768 ? 'High' :
                                     model.context_window >= 8192 ? 'Medium' : 'Basic'}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
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
            
            {/* No models found message */}
            {selectedProvider && models[selectedProvider] && getFilteredModels().length === 0 && (
              <div className="p-6 text-center bg-black/20 rounded-xl border border-white/20">
                <div className="text-gray-400 mb-2">🔍</div>
                <p className="text-gray-300 mb-2">No models found matching your filters</p>
                <button
                  onClick={() => {
                    setFilterQuality('all');
                    setFilterContext('all');
                  }}
                  className="text-blue-400 hover:text-blue-300 text-sm"
                >
                  Clear filters
                </button>
              </div>
            )}
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
                • Detailed reports<br/>
                • Large context windows
              </div>
            </div>
            <div className="text-center">
              <div className="text-blue-400 font-medium mb-1">Standard Models</div>
              <div className="text-gray-300">
                • Good balance<br/>
                • Fast processing<br/>
                • General research<br/>
                • Medium context windows
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400 font-medium mb-1">Basic Models</div>
              <div className="text-gray-300">
                • Ultra fast<br/>
                • Simple queries<br/>
                • Quick testing<br/>
                • Smaller context windows
              </div>
            </div>
          </div>
          
          {/* Context Window Comparison */}
          <div className="mt-4 p-3 bg-black/10 rounded-lg">
            <h4 className="text-white text-sm font-medium mb-2">Context Window Comparison</h4>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-2 text-xs">
              <div className="text-center">
                <div className="text-purple-400 font-medium">Ultra (131K+)</div>
                <div className="text-gray-400">Long documents, complex analysis</div>
              </div>
              <div className="text-center">
                <div className="text-yellow-400 font-medium">High (32K+)</div>
                <div className="text-gray-400">Detailed research, multiple sources</div>
              </div>
              <div className="text-center">
                <div className="text-blue-400 font-medium">Medium (8K+)</div>
                <div className="text-gray-400">Standard reports, summaries</div>
              </div>
              <div className="text-center">
                <div className="text-gray-400 font-medium">Basic (Any)</div>
                <div className="text-gray-400">Quick facts, simple queries</div>
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

      {/* Model Usage Statistics */}
      {selectedProvider && models[selectedProvider] && (
        <div className="bg-black/20 rounded-xl p-4 border border-white/20">
          <h3 className="text-white font-medium mb-3 flex items-center">
            <BarChart3 className="w-4 h-4 mr-2" />
            Model Statistics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400 mb-1">
                {Object.keys(models[selectedProvider]).length}
              </div>
              <div className="text-gray-300 text-xs">Total Models</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400 mb-1">
                {Object.values(models[selectedProvider]).filter(m => m.quality_tier === 'premium').length}
              </div>
              <div className="text-gray-300 text-xs">Premium Models</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400 mb-1">
                {Object.values(models[selectedProvider]).filter(m => m.active !== false).length}
              </div>
              <div className="text-gray-300 text-xs">Active Models</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400 mb-1">
                {Math.max(...Object.values(models[selectedProvider]).map(m => m.context_window || 0).filter(w => w > 0))}
              </div>
              <div className="text-gray-300 text-xs">Max Context</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Search Engine Selection Component
const SearchEngineSelector = ({ 
  selectedSearchEngine, 
  setSelectedSearchEngine, 
  isLoading 
}: {
  selectedSearchEngine: string;
  setSelectedSearchEngine: (engine: string) => void;
  isLoading: boolean;
}) => {
  return (
    <div className="space-y-3">
      <label className="block text-white text-sm font-medium mb-3">
        Search Engine
      </label>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div 
          className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
            selectedSearchEngine === 'free_web'
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-white/20 bg-black/10 hover:border-white/40'
          }`}
          onClick={() => !isLoading && setSelectedSearchEngine('free_web')}
        >
          <div className="flex items-center space-x-3">
            <div className="bg-yellow-500/10 p-2 rounded-lg">
              <SearchIcon className="w-5 h-5 text-yellow-400" />
            </div>
            <div>
              <h3 className="text-white font-medium">FusionSearch</h3>
              <p className="text-gray-300 text-sm">Combines Google & Bing for broad, reliable results</p>
            </div>
          </div>
        </div>

        <div 
          className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
            selectedSearchEngine === 'serpapi'
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-white/20 bg-black/10 hover:border-white/40'
          }`}
          onClick={() => !isLoading && setSelectedSearchEngine('serpapi')}
        >
          <div className="flex items-center space-x-3">
            <div className="bg-green-500/10 p-2 rounded-lg">
              <SearchIcon className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h3 className="text-white font-medium">SerpAPI</h3>
              <p className="text-gray-300 text-sm">Enhanced commercial results</p>
            </div>
          </div>
        </div>
      </div>
      
      {selectedSearchEngine === 'serpapi' && (
        <div className="flex items-center p-3 bg-yellow-500/10 rounded-xl border border-yellow-500/30">
          <AlertCircle className="w-4 h-4 text-yellow-400 mr-2" />
          <span className="text-yellow-300 text-sm">
            Requires SERPAPI_API_KEY in backend
          </span>
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
  const [selectedSearchEngine, setSelectedSearchEngine] = useState('free_web');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
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
      setIsLoadingModels(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/api/models`);
      const data = await response.json();
      
      if (data.success) {
        // Add three best models under 'together' and remove from 'nebius'
        data.models.nebius = {};
        data.models.together = {
          'meta-llama/Llama-3.2-3B-Instruct-Turbo': {
            name: 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
            provider: 'together',
            description: 'Meta Llama 3.2 3B Instruct Turbo - Cost-effective, good for basic task coordination and planning.',
            api_key_available: true,
            best_for: ['Basic Coordination', 'Planning', 'High Volume'],
            estimated_time: 'Fast',
            cost: '$0.06 per 1M tokens (~33.3M tokens for $2)',
            quality: 'standard',
            max_tokens: 8192,
            recommended_temperature: 0.2,
            quality_tier: 'basic',
            memory_usage: 'Cloud',
            id: 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
            owned_by: 'Meta',
            active: true,
            context_window: 8192,
            max_completion_tokens: 8192
          },
          'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free': {
            name: 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
            provider: 'together',
            description: 'DeepSeek R1 Distill Llama 70B (Free) - Specialized reasoning, excellent for complex planning. Free tier.',
            api_key_available: true,
            best_for: ['Complex Planning', 'Specialized Reasoning'],
            estimated_time: 'Medium',
            cost: 'Free',
            quality: 'premium',
            max_tokens: 8192,
            recommended_temperature: 0.3,
            quality_tier: 'premium',
            memory_usage: 'Cloud',
            id: 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
            owned_by: 'DeepSeek',
            active: true,
            context_window: 8192,
            max_completion_tokens: 8192
          },
          'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free': {
            name: 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            provider: 'together',
            description: 'Meta Llama 3.3 70B Instruct Turbo (Free) - High quality, free tier, great for advanced research.',
            api_key_available: true,
            best_for: ['Advanced Research', 'Complex Analysis'],
            estimated_time: 'Medium',
            cost: 'Free',
            quality: 'premium',
            max_tokens: 8192,
            recommended_temperature: 0.2,
            quality_tier: 'premium',
            memory_usage: 'Cloud',
            id: 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            owned_by: 'Meta',
            active: true,
            context_window: 8192,
            max_completion_tokens: 8192
          }
        };
        setModels(data.models);
        // Auto-select huggingface and DeepSeek-R1-0528 if huggingface is the only provider
        const providers = Object.keys(data.models);
        if (providers.length > 0) {
          const firstProvider = providers[0];
          setSelectedProvider(firstProvider);
          const modelKeys = Object.keys(data.models[firstProvider]);
          if (modelKeys.length > 0) {
            setSelectedModel(modelKeys[0]);
          }
        }
      } else {
        setError('Failed to load models from server');
      }
    } catch (err) {
      setError('Failed to load models. Make sure the backend is running.');
      console.error('Error loading models:', err);
    } finally {
      setIsLoadingModels(false);
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
        search_engine: selectedSearchEngine,
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
            <h1 className="text-4xl font-bold text-white">Chimera</h1>
          </div>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            A powerful AI research assistant that combines multiple AI models and search engines to generate comprehensive research reports. 
            Like the mythical chimera, it merges different capabilities into one unified research experience.
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
                <SearchIcon className="absolute left-4 top-4 w-5 h-5 text-gray-400" />
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

            {/* Query Analysis */}
            {query && (
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                <h3 className="text-blue-300 font-medium mb-2 flex items-center">
                  <Search className="w-4 h-4 mr-2" />
                  Query Analysis
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                  <div>
                    <span className="text-gray-400">Query Type: </span>
                    <span className="text-blue-300">
                      {query.toLowerCase().includes('code') || query.toLowerCase().includes('programming') ? 'Technical' :
                       query.toLowerCase().includes('research') || query.toLowerCase().includes('analysis') ? 'Academic' :
                       query.split(' ').length > 30 ? 'Complex' :
                       query.split(' ').length > 10 ? 'Detailed' : 'Simple'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Word Count: </span>
                    <span className="text-blue-300">{query.split(' ').length}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Recommended Context: </span>
                    <span className="text-blue-300">
                      {query.split(' ').length > 30 ? 'Large (32K+)' :
                       query.split(' ').length > 10 ? 'Medium (8K+)' : 'Small (Any)'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Model Selection */}
            {isLoadingModels ? (
              <div className="flex items-center justify-center p-8 bg-black/20 rounded-xl border border-white/20">
                <Loader2 className="w-6 h-6 animate-spin text-blue-400 mr-3" />
                <div>
                  <p className="text-white font-medium">Loading Models</p>
                  <p className="text-gray-400 text-sm">Fetching available AI models from providers...</p>
                </div>
              </div>
            ) : (
              <EnhancedModelSelector
                models={models}
                selectedProvider={selectedProvider}
                selectedModel={selectedModel}
                setSelectedProvider={setSelectedProvider}
                setSelectedModel={setSelectedModel}
                query={query}
                isLoading={isLoading}
              />
            )}

            {/* Search Engine Selection */}
            <SearchEngineSelector
              selectedSearchEngine={selectedSearchEngine}
              setSelectedSearchEngine={setSelectedSearchEngine}
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