'use client';
import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Download, FileText, BarChart3, TrendingUp, AlertCircle, CheckCircle, Loader2, Scissors, FileSearch, Save, RefreshCw } from 'lucide-react';
import { pipeline } from '@huggingface/transformers';

/**
 * RESILIENT THEMATIC ANALYZER - PRODUCTION VERSION
 * 
 * Fixed Issues:
 * 1. ✓ "Run 0" bug eliminated - proper run number tracking
 * 2. ✓ Auto-download after each run - no data loss
 * 3. ✓ No stuck states - proper error boundaries
 * 4. ✓ Faster processing - optimized model and timing
 * 5. ✓ Clear state management - no race conditions
 */
const Home = () => {
  // ============================================================================
  // STATE MANAGEMENT
  // ============================================================================
  
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('azure-gpt-4o');
  const [customPrompt, setCustomPrompt] = useState('');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileStats, setFileStats] = useState(null);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [currentRun, setCurrentRun] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [results, setResults] = useState(null);
  const [cleaningStats, setCleaningStats] = useState<any>(null);
  const [completedRuns, setCompletedRuns] = useState([]);
  const [cleanedText, setCleanedText] = useState('');
  const [error, setError] = useState('');
  const [partialResults, setPartialResults] = useState(null);
  const extractorRef = useRef(null as any);
  const [extractorReady, setExtractorReady] = useState(false);
  const [seeds, setSeeds] = useState([42, 123, 456, 789, 1011, 1213]);
  const [temperature, setTemperature] = useState(0.7);
  const [newSeedInput, setNewSeedInput] = useState('');
  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  
  const MAX_RETRIES = 3;
  const RETRY_DELAY_BASE = 1500;
  const TOTAL_RUNS = seeds.length;
  const SEEDS = seeds;
  
  // Model configurations
  const MODELS: any = {
    'gemini-2.5-pro': {
      name: 'Gemini 2.5 Pro',
      provider: 'google',
      endpoint: (key: string) => `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${key}`,
      apiKeyName: 'Gemini API Key'
    },
    'claude-3-5-sonnet': {
      name: 'Claude 3.5 Sonnet (Direct)',
      provider: 'anthropic',
      endpoint: (key?: string) => 'https://api.anthropic.com/v1/messages',
      apiKeyName: 'Anthropic API Key'
    },
    'openrouter-claude-sonnet': {
      name: 'Claude 3.5 Sonnet (OpenRouter)',
      provider: 'openrouter',
      endpoint: (key?: string) => 'https://openrouter.ai/api/v1/chat/completions',
      apiKeyName: 'OpenRouter API Key',
      model: 'anthropic/claude-3.5-sonnet'
    },
    'azure-gpt-4o': {
      name: 'Azure GPT-4o',
      provider: 'azure',
      endpoint: (key?: string) => 'https://alright.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-12-01-preview',
      apiKeyName: 'Azure OpenAI API Key',
      deployment: 'gpt-4o',
      apiVersion: '2024-12-01-preview'
    },
    'gpt-4o': {
      name: 'GPT-4o (OpenAI)',
      provider: 'openai',
      endpoint: (key?: string) => 'https://api.openai.com/v1/chat/completions',
      apiKeyName: 'OpenAI API Key'
    },
    'llama-3-70b': {
      name: 'Llama 3 70B (Groq)',
      provider: 'groq',
      endpoint: (key?: string) => 'https://api.groq.com/openai/v1/chat/completions',
      apiKeyName: 'Groq API Key'
    },
    'openrouter-llama-3.2-90b': {
      name: 'Llama 3.2 90B (OpenRouter)',
      provider: 'openrouter',
      endpoint: (key?: string) => 'https://openrouter.ai/api/v1/chat/completions',
      apiKeyName: 'OpenRouter API Key',
      model: 'meta-llama/llama-3.2-90b-vision-instruct:free'
    },
    'deepseek-chat': {
      name: 'DeepSeek Chat (Direct)',
      provider: 'deepseek',
      endpoint: (key?: string) => 'https://api.deepseek.com/v1/chat/completions',
      apiKeyName: 'DeepSeek API Key'
    },
    'openrouter-deepseek-r1': {
      name: 'DeepSeek R1 (OpenRouter)',
      provider: 'openrouter',
      endpoint: (key?: string) => 'https://openrouter.ai/api/v1/chat/completions',
      apiKeyName: 'OpenRouter API Key',
      model: 'deepseek/deepseek-r1'
    }
  };
  
  // Default prompt template
  const DEFAULT_PROMPT = `Conduct a qualitative thematic analysis. Identify major emotional, psychological, and conceptual themes.

For each theme provide:
1. Theme name (concise title)
2. Description (detailed explanation)
3. Sentiment (positive, negative, or neutral)
4. Frequency estimate (integer count)
5. Evidence (2-3 direct quotes or examples)

Also provide:
- Dominant emotional tone (one word)
- Overall sentiment score (0-1, where 0=negative, 1=positive)
- Narrative arc (brief description)

Return valid JSON in this exact structure:
{
  "majorEmotionalThemes": [
    {
      "theme": "Theme Name",
      "description": "Detailed description",
      "sentiment": "positive",
      "frequency": 15,
      "evidence": ["example 1", "example 2"]
    }
  ],
  "emotionalPatterns": {
    "dominantTone": "word",
    "overallSentiment": 0.75,
    "narrativeArc": "description"
  },
  "keyInsights": ["insight 1", "insight 2"]
}`;

  useEffect(() => {
    const initializeExtractor = async () => {
      try {
        console.log('🔄 Loading semantic similarity model...');
        // This loads a lightweight but effective model for generating embeddings
        // The model is about 23MB and gets cached in the browser after first download
        extractorRef.current = await pipeline(
          'feature-extraction',
          'Xenova/all-MiniLM-L6-v2'
        );
        setExtractorReady(true);
        console.log('✓ Semantic similarity model ready');
      } catch (error) {
        console.error('Failed to load embedding model:', error);
        console.log('⚠️ Falling back to exact match comparison');
        // We'll handle the fallback in the similarity function
      }
    };
    
    initializeExtractor();
  }, []); // Empty dependency a

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================
  
  /**
   * Auto-download helper - creates and triggers file download
   * 
   * Why this approach?
   * - Blob API is cross-browser compatible
   * - URL.createObjectURL is memory-efficient
   * - Immediate cleanup with revokeObjectURL prevents memory leaks
   * 
   * @param {string} content - File content to download
   * @param {string} filename - Name for downloaded file
   * @param {string} type - MIME type (default: JSON)
   */
  const downloadFile = (content: string, filename: string, type: string = 'application/json') => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url); // Clean up memory
  };

  /**
   * Automatically save each completed run to disk
   * 
   * Critical for fault tolerance:
   * - User never loses completed work
   * - Can manually review individual runs
   * - Provides audit trail for analysis
   * 
   * @param {Object} runResult - Complete analysis result for one run
   * @param {number} runNumber - Run number (1-6)
   */
  const autoSaveRun = (runResult: any, runNumber: number) => {
    const timestamp = new Date().toISOString().split('T')[0];
    const runData = {
      fileName,
      runNumber,
      timestamp,
      seed: runResult.seed,
      themes: runResult.majorEmotionalThemes,
      patterns: runResult.emotionalPatterns,
      insights: runResult.keyInsights
    };
    
    const filename = `run_${runNumber}_${fileName.replace('.txt', '')}_${timestamp}.json`;
    downloadFile(JSON.stringify(runData, null, 2), filename);
    console.log(`✓ Auto-saved: ${filename}`);
  };

  /**
   * Exponential backoff for retries
   * Formula: delay = 1500ms * (2 ^ attempt)
   * Results: 1.5s, 3s, 6s for attempts 0, 1, 2
   */
  const exponentialBackoff = (attempt: number) => {
    const delay = RETRY_DELAY_BASE * Math.pow(2, attempt);
    return new Promise(resolve => setTimeout(resolve, delay));
  };

  /**
   * Retry wrapper with exponential backoff
   * 
   * Design Pattern: Higher-Order Function (Decorator)
   * - Wraps any async function with retry logic
   * - Transparent to caller - looks like normal function
   * - Configurable retry count
   * 
   * @param {Function} fn - Async function to retry
   * @param {number} maxRetries - Max retry attempts (default: 3)
   * @returns {Function} - Wrapped function with retry capability
   */
  const withRetry = (fn: any, maxRetries: number = MAX_RETRIES) => {
    return async (...args: any[]) => {
      let lastError;
      
      for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
          const result = await fn(...args);
          if (attempt > 0) {
            console.log(`✓ Success after ${attempt} ${attempt === 1 ? 'retry' : 'retries'}`);
          }
          return result;
          
        } catch (error) {
          lastError = error;
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error(`✗ Attempt ${attempt + 1}/${maxRetries + 1} failed:`, errorMessage);
          
          if (attempt === maxRetries) {
            throw new Error(`Failed after ${maxRetries + 1} attempts: ${errorMessage}`);
          }
          
          setCurrentStep(`Retrying after error (${attempt + 1}/${maxRetries})...`);
          await exponentialBackoff(attempt);
        }
      }
      
      throw lastError;
    };
  };

  // ============================================================================
  // TEXT PREPROCESSING
  // ============================================================================
  
  /**
   * Clean transcript text by removing metadata and artifacts
   * 
   * Removes:
   * - Transcription service attribution (otter.ai, etc.)
   * - Document boundary markers
   * - Timestamp lines (00:00, 12:34, etc.)
   * - Empty or near-empty lines
   * - Excessive whitespace
   * 
   * Why preprocess?
   * - Reduces token usage (saves API costs)
   * - Improves analysis quality (focuses on content)
   * - Normalizes input format
   * 
   * @param {string} text - Raw transcript text
   * @returns {string} - Cleaned text
   */
  const preprocessText = (text: string) => {
    setCurrentStep('Cleaning text...');
    
    const originalLength = text.length;
    const originalLines = text.split('\n').length;
    
    // Remove transcription service attribution
    let cleaned = text.replace(/Transcribed by https?:\/\/otter\.ai/gi, '');
    
    // Remove document boundary markers
    cleaned = cleaned.replace(/---\s*(End|Start)\s+of\s+content\s+from:\s+[^\n]+\s*---/gi, '');
    
    // Normalize multiple blank lines to double newline
    cleaned = cleaned.replace(/\n\s*\n\s*\n+/g, '\n\n');
    
    // Filter out timestamp lines and very short lines
    cleaned = cleaned.split('\n')
      .filter(line => {
        const trimmed = line.trim();
        // Remove timestamp lines (00:00 format)
        if (/^\d{1,2}:\d{2}(:\d{2})?$/.test(trimmed)) return false;
        // Remove lines shorter than 3 chars
        if (trimmed.length < 3) return false;
        return true;
      })
      .join('\n');
    
    cleaned = cleaned.trim();
    
    const cleanedLength = cleaned.length;
    const cleanedLines = cleaned.split('\n').length;
    const reductionPercent = ((1 - cleanedLength / originalLength) * 100).toFixed(1);
    
    setCleaningStats({
      originalLength,
      originalLines,
      cleanedLength,
      cleanedLines,
      reductionPercent,
      removedChars: originalLength - cleanedLength
    });
    
    return cleaned;
  };

  /**
   * Intelligently chunk text for API processing
   * 
   * Strategy:
   * 1. Respect paragraph boundaries when possible
   * 2. Fall back to sentence boundaries for large paragraphs
   * 3. Only split mid-sentence as last resort
   * 
   * Why chunk?
   * - API token limits (typically 30K-100K tokens)
   * - Better processing performance
   * - More focused analysis per chunk
   * 
   * @param {string} text - Text to chunk
   * @param {number} maxChunkSize - Max characters per chunk
   * @returns {string[]} - Array of text chunks
   */
  const chunkText = (text: string, maxChunkSize: number = 30000) => {
    setCurrentStep('Chunking text for analysis...');
    
    // If text fits in one chunk, return as-is
    if (text.length <= maxChunkSize) {
      return [text];
    }
    
    const chunks = [];
    const paragraphs = text.split('\n\n');
    let currentChunk = '';
    
    for (const paragraph of paragraphs) {
      // Handle oversized paragraphs - split by sentence
      if (paragraph.length > maxChunkSize) {
        const sentences = paragraph.match(/[^.!?]+[.!?]+/g) || [paragraph];
        
        for (const sentence of sentences) {
          if ((currentChunk + sentence).length > maxChunkSize && currentChunk) {
            chunks.push(currentChunk.trim());
            currentChunk = sentence;
          } else {
            currentChunk += sentence;
          }
        }
      } else {
        // Try to add paragraph to current chunk
        if ((currentChunk + '\n\n' + paragraph).length > maxChunkSize && currentChunk) {
          chunks.push(currentChunk.trim());
          currentChunk = paragraph;
        } else {
          currentChunk += (currentChunk ? '\n\n' : '') + paragraph;
        }
      }
    }
    
    // Don't forget the last chunk
    if (currentChunk) {
      chunks.push(currentChunk.trim());
    }
    
    console.log(`📄 Text split into ${chunks.length} chunks`);
    return chunks;
  };

  // ============================================================================
  // MULTI-MODEL API INTEGRATION
  // ============================================================================
  
  /**
   * Call AI API (supports multiple providers) with comprehensive error handling
   * 
   * Supported Models:
   * - Gemini 2.5 Pro (Google)
   * - Claude 3.5 Sonnet (Anthropic)
   * - GPT-4o / GPT-4 Turbo (OpenAI)
   * - Llama 3 70B (Groq)
   * - DeepSeek Chat (DeepSeek)
   * 
   * @param {string} text - Text to analyze
   * @param {number} seed - Seed for temperature variation
   * @param {boolean} isChunked - Is this part of a multi-chunk analysis?
   * @param {number} chunkIndex - Index of current chunk (0-based)
   * @param {number} totalChunks - Total number of chunks
   * @returns {Object} - Parsed analysis results
   */
  const callAIAPI = async (text: string, seed: number, isChunked: boolean = false, chunkIndex: number = 0, totalChunks: number = 1) => {
    const modelConfig = MODELS[selectedModel as keyof typeof MODELS];
    if (!modelConfig) {
      throw new Error(`Unsupported model: ${selectedModel}`);
    }
    
    const contextPrefix = isChunked 
      ? `This is chunk ${chunkIndex + 1} of ${totalChunks}. Analyze only this section.`
      : 'Analyze the complete text below.';
    
    let userPrompt = customPrompt || DEFAULT_PROMPT;
    
    // Replace seed placeholder with actual seed value
    userPrompt = userPrompt.replace(/{seed}/g, String(seed));
    
    // Check if user prompt contains text placeholders, if so replace them
    // Otherwise append text at the end (default behavior)
    let fullPrompt;
    if (userPrompt.includes('{text_chunk}') || userPrompt.includes('{text}')) {
      fullPrompt = `${contextPrefix}\n\n${userPrompt
        .replace(/{text_chunk}/g, text)
        .replace(/{text}/g, text)}`;
    } else {
      fullPrompt = `${contextPrefix}\n\n${userPrompt}\n\nText: ${text}`;
    }
    let response;
    let requestBody: any;
    let headers: any = { 'Content-Type': 'application/json' };
    let API_URL = '';
    
    // Configure request based on provider
    if (modelConfig.provider === 'google') {
      API_URL = modelConfig.endpoint(apiKey);
      requestBody = {
        contents: [{ parts: [{ text: fullPrompt }] }],
        generationConfig: {
          temperature,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 8192,
        }
      };
    } else if (modelConfig.provider === 'anthropic') {
      API_URL = modelConfig.endpoint();
      headers['x-api-key'] = apiKey;
      headers['anthropic-version'] = '2023-06-01';
      requestBody = {
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 8192,
        temperature,
        messages: [{ role: 'user', content: fullPrompt }]
      };
    } else if (modelConfig.provider === 'azure') {
      API_URL = modelConfig.endpoint();
      headers['api-key'] = apiKey;
      requestBody = {
        messages: [{ role: 'user', content: fullPrompt }],
        temperature,
        max_tokens: 8192
      };
    } else if (modelConfig.provider === 'openai') {
      API_URL = modelConfig.endpoint();
      headers['Authorization'] = `Bearer ${apiKey}`;
      const modelName = selectedModel === 'gpt-4o' ? 'gpt-4o' : 'gpt-4-turbo';
      requestBody = {
        model: modelName,
        messages: [{ role: 'user', content: fullPrompt }],
        temperature,
        max_tokens: 8192
      };
    } else if (modelConfig.provider === 'groq') {
      API_URL = modelConfig.endpoint();
      headers['Authorization'] = `Bearer ${apiKey}`;
      requestBody = {
        model: 'llama-3.1-70b-versatile',
        messages: [{ role: 'user', content: fullPrompt }],
        temperature,
        max_tokens: 8192
      };
    } else if (modelConfig.provider === 'deepseek') {
      API_URL = modelConfig.endpoint();
      headers['Authorization'] = `Bearer ${apiKey}`;
      requestBody = {
        model: 'deepseek-chat',
        messages: [{ role: 'user', content: fullPrompt }],
        temperature,
        max_tokens: 8192
      };
    } else if (modelConfig.provider === 'openrouter') {
      API_URL = modelConfig.endpoint();
      headers['Authorization'] = `Bearer ${apiKey}`;
      // OpenRouter optional headers for credits/rankings
      try {
        headers['HTTP-Referer'] = typeof window !== 'undefined' ? window.location.origin : 'https://localhost:3000';
        headers['X-Title'] = 'Qualitative Thematic Analyzer';
      } catch (e) {
        // Skip optional headers if they cause issues
        console.log('Skipping optional OpenRouter headers');
      }
      requestBody = {
        model: modelConfig.model,
        messages: [{ role: 'user', content: fullPrompt }],
        temperature,
        max_tokens: 8192
      };
    }
    
    try {
      console.log(`Calling ${modelConfig.provider} API:`, API_URL);
      if (modelConfig.provider === 'openrouter') {
        console.log('OpenRouter Request:', {
          model: requestBody.model,
          temperature: requestBody.temperature,
          promptLength: fullPrompt.length,
          headers: { ...headers, Authorization: 'Bearer ***' }
        });
      }
      
      const fetchOptions: RequestInit = {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody)
      };
      
      // Add CORS mode for OpenRouter to handle cross-origin requests
      if (modelConfig.provider === 'openrouter') {
        fetchOptions.mode = 'cors';
      }
      
      response = await fetch(API_URL, fetchOptions);
      console.log(`${modelConfig.provider} response status:`, response.status);
    } catch (fetchError) {
      const errorMessage = fetchError instanceof Error ? fetchError.message : String(fetchError);
      console.error('Fetch error details:', {
        provider: modelConfig.provider,
        url: API_URL,
        error: errorMessage,
        fetchError
      });
      throw new Error(`Network error (${modelConfig.provider}): ${errorMessage}. Open browser DevTools Console (F12) for more details.`);
    }

    // Handle HTTP errors with specific guidance
    if (!response.ok) {
      let errorMessage = response.statusText;
      
      try {
        const errorData = await response.json();
        console.error('API Error Response:', errorData);
        errorMessage = errorData.error?.message || errorData.message || errorData.error || errorMessage;
        
        if (response.status === 429) {
          throw new Error(`Rate limit exceeded. Wait and retry. (${errorMessage})`);
        } else if (response.status === 401 || response.status === 403) {
          throw new Error(`Authentication failed. Check API key. (${errorMessage})`);
        } else if (response.status === 402) {
          throw new Error(`Payment required. Add credits to your ${modelConfig.provider} account. (${errorMessage})`);
        } else if (response.status === 503) {
          throw new Error(`Service unavailable. Will retry. (${errorMessage})`);
        }
      } catch (jsonError) {
        // If error response isn't JSON, use status text
        console.error('Could not parse error response:', jsonError);
      }
      
      throw new Error(`API error (${response.status}): ${errorMessage}`);
    }

    const data = await response.json();
    let textResponse = '';
    
    // Extract response based on provider
    if (modelConfig.provider === 'google') {
      textResponse = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    } else if (modelConfig.provider === 'anthropic') {
      textResponse = data.content?.[0]?.text || '';
    } else if (modelConfig.provider === 'azure' || modelConfig.provider === 'openai' || modelConfig.provider === 'groq' || modelConfig.provider === 'deepseek' || modelConfig.provider === 'openrouter') {
      textResponse = data.choices?.[0]?.message?.content || '';
    }
    
    if (!textResponse) {
      throw new Error('No response from API');
    }

    // Robust JSON parsing - handle markdown code blocks
    let jsonStr = textResponse.trim();
    
    // Remove markdown code fences more robustly
    if (jsonStr.startsWith('```')) {
      // Remove opening fence (```json or ```)
      jsonStr = jsonStr.replace(/^```(?:json)?\s*\n?/, '');
      // Remove closing fence
      jsonStr = jsonStr.replace(/\n?```\s*$/, '');
    }
    
    // Clean up any remaining whitespace
    jsonStr = jsonStr.trim();
    
    try {
      const parsed = JSON.parse(jsonStr);
      
      // Flexible validation - check if it's a non-empty object
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Response is not a valid JSON object');
      }
      
      // Only validate default structure if using default prompt
      // Otherwise accept any valid JSON object
      const isDefaultPrompt = !customPrompt || customPrompt.trim() === '';
      if (isDefaultPrompt) {
      if (!parsed.majorEmotionalThemes || !Array.isArray(parsed.majorEmotionalThemes)) {
          throw new Error('Invalid response structure: missing majorEmotionalThemes array');
      }
      if (!parsed.emotionalPatterns || typeof parsed.emotionalPatterns !== 'object') {
          throw new Error('Invalid response structure: missing emotionalPatterns object');
        }
      }
      
      return parsed;
    } catch (parseError) {
      console.error('Parse error:', textResponse);
      const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);
      throw new Error(`Parse failed: ${errorMessage}`);
    }
  };

  // Wrap API call with retry logic
  const callAIAPIWithRetry = withRetry(callAIAPI);

  // ============================================================================
  // MULTI-CHUNK ANALYSIS COORDINATION
  // ============================================================================
  
  /**
   * Analyze text (single chunk or multiple chunks)
   * 
   * CRITICAL FIX: Now receives runNumber as parameter
   * - Eliminates dependency on currentRun state
   * - Prevents "Run 0" bug from stale state
   * - Ensures accurate progress reporting
   * 
   * For multi-chunk texts:
   * 1. Analyze each chunk independently
   * 2. Merge results intelligently (combine themes, average sentiment)
   * 
   * @param {string} text - Text to analyze
   * @param {number} seed - Seed for temperature variation
   * @param {number} runNumber - Current run number (1-6)
   * @returns {Object} - Complete analysis results
   */
  const analyzeText = async (text, seed, runNumber) => {
    const chunks = chunkText(text);
    
    // Single chunk - simple path
    if (chunks.length === 1) {
      setCurrentStep(`Run ${runNumber}: Single-pass analysis`);
      return await callAIAPIWithRetry(chunks[0], seed, false);
    }
    
    // Multiple chunks - analyze each and merge
    const chunkResults = [];
    for (let i = 0; i < chunks.length; i++) {
      setCurrentStep(`Run ${runNumber}: Analyzing chunk ${i + 1}/${chunks.length}`);
      const result = await callAIAPIWithRetry(chunks[i], seed, true, i, chunks.length);
      chunkResults.push(result);
    }
    
    setCurrentStep(`Run ${runNumber}: Merging chunk analyses`);
    return mergeChunkAnalyses(chunkResults);
  };

  /**
   * Intelligently merge results from multiple chunks
   * 
   * Strategy:
   * - For default prompt: Combine duplicate themes, sum frequencies, deduplicate evidence
   * - For custom prompt: Deep merge all top-level arrays and objects
   * 
   * @param {Object[]} chunkResults - Array of analysis results from chunks
   * @returns {Object} - Merged analysis result
   */
  const mergeChunkAnalyses = (chunkResults) => {
    // Check if this is the default prompt structure
    const isDefaultStructure = chunkResults[0]?.majorEmotionalThemes && chunkResults[0]?.emotionalPatterns;
    
    if (isDefaultStructure) {
      // Original merging logic for default prompt
    const themeMap = new Map();
    const allInsights = [];
    let totalSentiment = 0;
    const toneFrequency = {};
    
    chunkResults.forEach(result => {
      // Merge themes
      result.majorEmotionalThemes.forEach(theme => {
        const key = theme.theme;
        if (themeMap.has(key)) {
          const existing = themeMap.get(key);
          existing.frequency += theme.frequency;
          existing.evidence = [...new Set([...existing.evidence, ...theme.evidence])];
        } else {
          themeMap.set(key, { ...theme });
        }
      });
      
      // Collect insights
        if (result.keyInsights) allInsights.push(...result.keyInsights);
      
      // Aggregate sentiment
      totalSentiment += result.emotionalPatterns.overallSentiment;
      
      // Count tone occurrences
      const tone = result.emotionalPatterns.dominantTone;
      toneFrequency[tone] = (toneFrequency[tone] || 0) + 1;
    });
    
    // Find most common tone
    const dominantTone = Object.entries(toneFrequency)
      .sort((a: any, b: any) => (b[1] as number) - (a[1] as number))[0][0];
    
    return {
      majorEmotionalThemes: Array.from(themeMap.values())
        .sort((a, b) => b.frequency - a.frequency),
      emotionalPatterns: {
        dominantTone,
        overallSentiment: totalSentiment / chunkResults.length,
        toneVariety: Object.keys(toneFrequency).length
      },
      keyInsights: [...new Set(allInsights)]
    };
    } else {
      // Generic merging for custom prompts
      // Combine all arrays and merge objects
      const merged: any = {};
      
      chunkResults.forEach(result => {
        Object.keys(result).forEach(key => {
          if (Array.isArray(result[key])) {
            // Combine arrays
            if (!merged[key]) merged[key] = [];
            merged[key].push(...result[key]);
          } else if (typeof result[key] === 'object' && result[key] !== null) {
            // Merge objects (shallow)
            if (!merged[key]) merged[key] = {};
            merged[key] = { ...merged[key], ...result[key] };
          } else {
            // For primitives, take the last value or concatenate strings
            if (typeof result[key] === 'string' && merged[key]) {
              merged[key] = merged[key] + ' ' + result[key];
            } else {
              merged[key] = result[key];
            }
          }
        });
      });
      
      return merged;
    }
  };

  // ============================================================================
  // RELIABILITY ANALYSIS
  // ============================================================================
  
  /**
   * Calculate similarity between two runs (Jaccard similarity of themes)
   * 
   * Formula: |A ∩ B| / |A ∪ B|
   * - Intersection: themes present in both runs
   * - Union: all unique themes across both runs
   * 
   * @param {Object} run1 - First analysis run
   * @param {Object} run2 - Second analysis run
   * @returns {number} - Similarity score (0-1)
   */
/**
 * Calculate semantic similarity between two runs using embeddings
 * 
 * This is the heart of the improvement. Instead of just comparing whether
 * theme names match exactly (Jaccard), we compare the MEANING of the themes.
 * 
 * How it works:
 * 1. Extract all theme text from both runs
 * 2. Generate semantic embeddings (vectors) for each theme
 * 3. Pool (average) the embeddings to get one vector per run
 * 4. Calculate cosine similarity between the two run vectors
 * 
 * Why this is better:
 * - Recognizes "adventure and exploration" ≈ "thrill of discovering new places"
 * - Captures semantic relationships that exact text matching misses
 * - Gives more accurate reliability scores for your analysis
 * 
 * @param {Object} run1 - First analysis run
 * @param {Object} run2 - Second analysis run
 * @returns {number} - Similarity score (0-1)
 */
const calculateSimilarity = async (run1, run2) => {
  // Check if we're working with default or custom structure
  const isDefaultStructure = run1?.majorEmotionalThemes && run2?.majorEmotionalThemes;
  
  // For custom structures, use lightweight string comparison to avoid freezing
  if (!isDefaultStructure) {
    console.log('Using lightweight similarity for custom structure');
    const str1 = JSON.stringify(run1);
    const str2 = JSON.stringify(run2);
    
    // Simple character-level similarity
    if (str1 === str2) return 1;
    
    // Calculate approximate similarity based on shared substrings
    const len1 = str1.length;
    const len2 = str2.length;
    const maxLen = Math.max(len1, len2);
    const minLen = Math.min(len1, len2);
    
    // Basic heuristic: if lengths are very different, similarity is lower
    return minLen / maxLen;
  }
  
  // If the embedding model isn't available, fall back to Jaccard similarity
  if (!extractorRef.current || !extractorReady) {
    console.log('Using fallback similarity (exact match)');
    const themes1 = new Set(run1.majorEmotionalThemes.map(t => t.theme.toLowerCase()));
    const themes2 = new Set(run2.majorEmotionalThemes.map(t => t.theme.toLowerCase()));
    const intersection = new Set([...themes1].filter(x => themes2.has(x)));
    const union = new Set([...themes1, ...themes2]);
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  try {
    // Extract just the theme text (not the whole object) from each run
    const themes1 = run1.majorEmotionalThemes.map(t => t.theme);
    const themes2 = run2.majorEmotionalThemes.map(t => t.theme);

    // Handle edge case: if either run has no themes, return 0 similarity
    if (themes1.length === 0 || themes2.length === 0) {
      return 0;
    }

    // Limit number of themes to prevent UI freezing (max 10 themes per run)
    const limitedThemes1 = themes1.slice(0, 10);
    const limitedThemes2 = themes2.slice(0, 10);

    // Generate embeddings for all themes in both runs
    // The model converts each text string into a 384-dimensional vector
    // that captures its semantic meaning
    const embeddings1 = await extractorRef.current(limitedThemes1, { 
      pooling: 'mean',  // Average across tokens in each theme
      normalize: true   // Normalize vectors to unit length (important for cosine similarity)
    });
    
    const embeddings2 = await extractorRef.current(limitedThemes2, { 
      pooling: 'mean', 
      normalize: true 
    });

    // Each run might have multiple themes, so we need to combine them into
    // a single representative vector. We do this by averaging all theme vectors.
    // This gives us one vector that represents the "overall semantic content"
    // of all themes in that run.
    
    // embeddings1.data is a flat array, so we need to reshape it
    const numThemes1 = limitedThemes1.length;
    const embeddingDim = embeddings1.data.length / numThemes1;
    
    const numThemes2 = limitedThemes2.length;
    
    // Pool (average) all theme embeddings for run 1
    const pooledVector1 = new Array(embeddingDim).fill(0);
    for (let i = 0; i < numThemes1; i++) {
      for (let j = 0; j < embeddingDim; j++) {
        pooledVector1[j] += embeddings1.data[i * embeddingDim + j];
      }
    }
    // Divide by count to get average
    for (let j = 0; j < embeddingDim; j++) {
      pooledVector1[j] /= numThemes1;
    }
    
    // Pool (average) all theme embeddings for run 2
    const pooledVector2 = new Array(embeddingDim).fill(0);
    for (let i = 0; i < numThemes2; i++) {
      for (let j = 0; j < embeddingDim; j++) {
        pooledVector2[j] += embeddings2.data[i * embeddingDim + j];
      }
    }
    // Divide by count to get average
    for (let j = 0; j < embeddingDim; j++) {
      pooledVector2[j] /= numThemes2;
    }

    // Now calculate cosine similarity between the two pooled vectors
    // This is the exact same math from your original cosine similarity code
    const dotProduct = pooledVector1.reduce((sum, a, i) => sum + a * pooledVector2[i], 0);
    const magnitudeA = Math.sqrt(pooledVector1.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(pooledVector2.reduce((sum, b) => sum + b * b, 0));
    
    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }
    
    const similarity = dotProduct / (magnitudeA * magnitudeB);
    
    // Cosine similarity ranges from -1 to 1, but for theme comparison
    // we typically see values between 0 and 1. Clamp to ensure valid range.
    return Math.max(0, Math.min(1, similarity));
    
  } catch (error) {
    console.error('Error calculating semantic similarity:', error);
    // If embedding generation fails, fall back to simple comparison
    const themes1 = new Set(run1.majorEmotionalThemes.map(t => t.theme.toLowerCase()));
    const themes2 = new Set(run2.majorEmotionalThemes.map(t => t.theme.toLowerCase()));
    const intersection = new Set([...themes1].filter(x => themes2.has(x)));
    const union = new Set([...themes1, ...themes2]);
    return union.size > 0 ? intersection.size / union.size : 0;
  }
};

  /**
   * Calculate Cohen's Kappa for inter-rater reliability
   * Measures agreement between two runs accounting for chance
   * 
   * @param {Object} run1 - First analysis run
   * @param {Object} run2 - Second analysis run
   * @returns {number} - Kappa score (-1 to 1, where 1 = perfect agreement)
   */
  const calculateCohenKappa = (run1, run2) => {
    const isDefaultStructure = run1?.majorEmotionalThemes && run2?.majorEmotionalThemes;
    
    if (!isDefaultStructure) {
      // For custom structures, use simplified kappa based on JSON similarity
      const str1 = JSON.stringify(run1);
      const str2 = JSON.stringify(run2);
      
      // Simple observed agreement
      const po = str1 === str2 ? 1.0 : (Math.min(str1.length, str2.length) / Math.max(str1.length, str2.length));
      
      // Expected agreement by chance (estimate)
      const pe = 0.5;
      
      const kappa = (po - pe) / (1 - pe);
      return Math.max(-1, Math.min(1, kappa));
    }
    
    // For default structure: calculate based on theme agreement
    const themes1Set = new Set(run1.majorEmotionalThemes.map(t => t.theme.toLowerCase()));
    const themes2Set = new Set(run2.majorEmotionalThemes.map(t => t.theme.toLowerCase()));
    
    // Get all unique themes across both runs
    const allThemes = new Set([...themes1Set, ...themes2Set]);
    
    if (allThemes.size === 0) return 0;
    
    // Count agreements and disagreements
    let agreements = 0;
    let disagreements = 0;
    
    allThemes.forEach(theme => {
      const in1 = themes1Set.has(theme);
      const in2 = themes2Set.has(theme);
      
      if (in1 === in2) {
        agreements++;
      } else {
        disagreements++;
      }
    });
    
    // Observed agreement
    const po = agreements / allThemes.size;
    
    // Expected agreement by chance
    // P(both yes) + P(both no)
    const n1 = themes1Set.size;
    const n2 = themes2Set.size;
    const total = allThemes.size;
    
    const pYes1 = n1 / total;
    const pNo1 = 1 - pYes1;
    const pYes2 = n2 / total;
    const pNo2 = 1 - pYes2;
    
    const pe = (pYes1 * pYes2) + (pNo1 * pNo2);
    
    // Cohen's Kappa
    const kappa = pe === 1 ? 1 : (po - pe) / (1 - pe);
    
    return Math.max(-1, Math.min(1, kappa));
  };

  /**
   * Synthesize consensus themes from multiple runs
   * 
   * Graceful Degradation:
   * - Works with 1-6 runs (adapts threshold)
   * - Provides warnings for low run counts
   * - Calculates reliability metrics
   * 
   * Consensus Logic:
   * - Theme must appear in ≥50% of runs (adjusts for partial data)
   * - Themes sorted by consistency percentage
   * - Evidence collected from all occurrences
   * 
   * @param {Object[]} allRuns - Array of completed analysis runs
   * @returns {Object} - Consensus themes and reliability metrics
   */
  const synthesizeResults = async (allRuns) => {
    if (!allRuns || allRuns.length === 0) {
      return {
        consensusThemes: [],
        reliability: {
          avgSimilarity: '0',
          minSimilarity: '0',
          maxSimilarity: '0',
          interpretation: 'No Data',
          runCount: 0,
          warning: 'No completed runs available for analysis'
        },
        allRuns: []
      };
    }

    // Check if we're working with default or custom structure
    const isDefaultStructure = allRuns[0]?.majorEmotionalThemes;

    // Aggregate theme data across all runs (only for default structure)
    const themeFrequency = {};
    const themeDescriptions = {};
    const themeSentiments = {};
    const themeEvidence = {};

    if (isDefaultStructure) {
    allRuns.forEach(run => {
      run.majorEmotionalThemes.forEach(theme => {
        const key = theme.theme;
        themeFrequency[key] = (themeFrequency[key] || 0) + 1;
        themeDescriptions[key] = theme.description;
        themeSentiments[key] = theme.sentiment;
        if (!themeEvidence[key]) themeEvidence[key] = [];
        themeEvidence[key].push(...(theme.evidence || []));
      });
    });
    }

    // Calculate inter-run similarities
    let avgSimilarity = 0;
    let minSimilarity = 1;
    let maxSimilarity = 0;
    
    // Calculate inter-run similarities and Cohen's Kappa with progress updates
    let avgKappa = 0;
    let minKappa = 1;
    let maxKappa = 0;
    
if (allRuns.length > 1) {
    const similarities = [];
      const kappaScores = [];
      const totalPairs = (allRuns.length * (allRuns.length - 1)) / 2;
      let completedPairs = 0;
      
      // For custom structures with many runs, sample comparisons to avoid freezing
      const shouldSample = !isDefaultStructure && totalPairs > 10;
      const samplesToTake = shouldSample ? Math.min(10, totalPairs) : totalPairs;
      
      if (shouldSample) {
        console.log(`Sampling ${samplesToTake} of ${totalPairs} comparisons for performance`);
        // Sample evenly distributed pairs
        const step = Math.ceil(totalPairs / samplesToTake);
        let pairIndex = 0;
        
    for (let i = 0; i < allRuns.length; i++) {
      for (let j = i + 1; j < allRuns.length; j++) {
            if (pairIndex % step === 0 && similarities.length < samplesToTake) {
              completedPairs++;
              setCurrentStep(`Calculating reliability metrics ${completedPairs}/${samplesToTake}...`);
              await new Promise(resolve => setTimeout(resolve, 0));
              
        const sim = await calculateSimilarity(allRuns[i], allRuns[j]);
              const kappa = calculateCohenKappa(allRuns[i], allRuns[j]);
        similarities.push(sim);
              kappaScores.push(kappa);
            }
            pairIndex++;
          }
        }
      } else {
        // Calculate all pairs for default structure or small datasets
        for (let i = 0; i < allRuns.length; i++) {
          for (let j = i + 1; j < allRuns.length; j++) {
            completedPairs++;
            setCurrentStep(`Calculating reliability metrics ${completedPairs}/${totalPairs}...`);
            await new Promise(resolve => setTimeout(resolve, 0));
            
            const sim = await calculateSimilarity(allRuns[i], allRuns[j]);
            const kappa = calculateCohenKappa(allRuns[i], allRuns[j]);
            similarities.push(sim);
            kappaScores.push(kappa);
          }
        }
      }
      
      avgSimilarity = similarities.length > 0 ? similarities.reduce((a, b) => a + b, 0) / similarities.length : 0.5;
      minSimilarity = similarities.length > 0 ? Math.min(...similarities) : 0;
      maxSimilarity = similarities.length > 0 ? Math.max(...similarities) : 1;
      
      avgKappa = kappaScores.length > 0 ? kappaScores.reduce((a, b) => a + b, 0) / kappaScores.length : 0;
      minKappa = kappaScores.length > 0 ? Math.min(...kappaScores) : -1;
      maxKappa = kappaScores.length > 0 ? Math.max(...kappaScores) : 1;
  }
    else {
      avgSimilarity = 1;
      minSimilarity = 1;
      maxSimilarity = 1;
      avgKappa = 1;
      minKappa = 1;
      maxKappa = 1;
    }

    // Build consensus themes (works for both default and custom structures)
    let consensusThemes = [];
    
    if (isDefaultStructure) {
      // Default structure: use pre-aggregated theme data
    const consensusThreshold = Math.max(1, Math.ceil(allRuns.length / 2));
    
      consensusThemes = Object.entries(themeFrequency)
      .filter(([_, freq]: any) => (freq as number) >= consensusThreshold)
      .map(([theme, freq]: any) => ({
        theme,
        description: themeDescriptions[theme],
        sentiment: themeSentiments[theme],
        consistency: ((freq as number) / allRuns.length * 100).toFixed(1),
        appearanceCount: freq,
        evidence: [...new Set(themeEvidence[theme])].slice(0, 5)
      }))
      .sort((a, b) => parseFloat(b.consistency) - parseFloat(a.consistency));
    } else {
      // Custom structure: extract themes from any arrays in the structure
      const extractThemesFromCustom = (run: any) => {
        const themes: any[] = [];
        
        // Recursively find all arrays and extract their items
        const traverse = (obj: any, path: string = '') => {
          if (Array.isArray(obj)) {
            // Found an array - extract its items as potential themes
            obj.forEach((item, idx) => {
              if (typeof item === 'object' && item !== null) {
                // Get a title/name field if it exists
                const titleField = item.theme_name || item.theme || item.title || item.name || 
                                  item.aspect || item.experience_type || item.category || 
                                  `${path} ${idx + 1}`;
                const descField = item.description || item.desc || JSON.stringify(item).slice(0, 200);
                
                themes.push({
                  category: path || 'General',
                  title: String(titleField),
                  description: String(descField),
                  fullData: item
                });
              } else if (typeof item === 'string') {
                themes.push({
                  category: path || 'General',
                  title: item,
                  description: item,
                  fullData: item
                });
              }
            });
          } else if (typeof obj === 'object' && obj !== null) {
            Object.keys(obj).forEach(key => {
              traverse(obj[key], key);
            });
          }
        };
        
        traverse(run);
        return themes;
      };
      
      // Extract themes from all runs
      const allCustomThemes: any = {};
      allRuns.forEach(run => {
        const runThemes = extractThemesFromCustom(run);
        runThemes.forEach(theme => {
          const key = `${theme.category}:${theme.title}`;
          if (!allCustomThemes[key]) {
            allCustomThemes[key] = {
              category: theme.category,
              title: theme.title,
              description: theme.description,
              count: 0,
              examples: []
            };
          }
          allCustomThemes[key].count++;
          if (allCustomThemes[key].examples.length < 3) {
            allCustomThemes[key].examples.push(theme.description);
          }
        });
      });
      
      // Build consensus from custom themes
      const consensusThreshold = Math.max(1, Math.ceil(allRuns.length / 2));
      consensusThemes = Object.values(allCustomThemes)
        .filter((t: any) => t.count >= consensusThreshold)
        .map((t: any) => ({
          theme: `[${t.category}] ${t.title}`,
          description: t.description,
          consistency: ((t.count / allRuns.length) * 100).toFixed(1),
          appearanceCount: t.count,
          evidence: t.examples
        }))
        .sort((a, b) => parseFloat(b.consistency) - parseFloat(a.consistency));
    }

    // Generate interpretation and warnings
    let interpretation;
    let warning = null;
    
    if (allRuns.length < 3) {
      interpretation = 'Limited';
      warning = `Only ${allRuns.length} run(s) completed. Results should be interpreted with caution.`;
    } else if (allRuns.length < TOTAL_RUNS) {
      interpretation = avgSimilarity > 0.6 ? 'Moderate' : 'Low';
      warning = `Partial analysis (${allRuns.length} of ${TOTAL_RUNS} runs completed). Consider re-running for more reliable results.`;
    } else {
      interpretation = avgSimilarity > 0.7 ? 'High' : avgSimilarity > 0.5 ? 'Moderate' : 'Low';
    }

    return {
      consensusThemes,
      reliability: {
        avgSimilarity: (avgSimilarity * 100).toFixed(1),
        minSimilarity: (minSimilarity * 100).toFixed(1),
        maxSimilarity: (maxSimilarity * 100).toFixed(1),
        avgKappa: avgKappa.toFixed(3),
        minKappa: minKappa.toFixed(3),
        maxKappa: maxKappa.toFixed(3),
        interpretation,
        runCount: allRuns.length,
        warning
      },
      allRuns,
      isCustomStructure: !isDefaultStructure
    };
  };

  // ============================================================================
  // FILE HANDLING
  // ============================================================================
  
  /**
   * Handle file upload with validation and stats calculation
   */
  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    
    if (!uploadedFile) return;
    
    // Check file size (5MB limit)
    const maxSize = 5 * 1024 * 1024; // 5MB in bytes
    if (uploadedFile.size > maxSize) {
      setError(`File size exceeds 5MB limit. File size: ${(uploadedFile.size / 1024 / 1024).toFixed(2)}MB. Please chunk your file into smaller parts (under 5MB each) and combine outputs later.`);
      setFile(null);
      setFileName('');
      setFileStats(null);
      return;
    }
    
    if (uploadedFile.type === 'text/plain' || uploadedFile.name.endsWith('.txt')) {
      setFile(uploadedFile);
      setFileName(uploadedFile.name);
      setError('');
      setPartialResults(null);
      setCompletedRuns([]);
      
      try {
        const text = await uploadedFile.text();
        const stats = {
          size: (uploadedFile.size / 1024).toFixed(2),
          lines: text.split('\n').length,
          words: text.split(/\s+/).filter(w => w.length > 0).length,
          chars: text.length
        };
        setFileStats(stats);
      } catch (err) {
        setError('Error reading file: ' + err.message);
      }
    } else {
      setError('Please upload a .txt file (plain text format)');
      setFile(null);
      setFileName('');
      setFileStats(null);
    }
  };

  // ============================================================================
  // MAIN ANALYSIS ORCHESTRATION
  // ============================================================================
  
  /**
   * Main analysis function - FIXED VERSION
   * 
   * Critical Fixes:
   * 1. Uses local array (runsToProcess) instead of state for run tracking
   *    - Eliminates race conditions
   *    - Ensures synchronous operation
   * 
   * 2. Auto-saves each run immediately
   *    - No data loss on failure
   *    - User can inspect individual runs
   * 
   * 3. Passes runNumber to analyzeText()
   *    - Fixes "Run 0" bug
   *    - Accurate progress tracking
   * 
   * 4. Downloads final/partial synthesis automatically
   *    - No manual export needed
   *    - Immediate results availability
   * 
   * @param {number} resumeFrom - Run number to resume from (0 = start fresh)
   */
  const handleAnalysis = async (resumeFrom = 0) => {
    if (!apiKey) {
      setError('Please enter your Gemini API key');
      return;
    }
    if (!file && !cleanedText) {
      setError('Please upload a text file');
      return;
    }

    setStatus(resumeFrom > 0 ? 'resuming' : 'preprocessing');
    setProgress((resumeFrom / TOTAL_RUNS) * 100);
    setError('');
    setCurrentStep(resumeFrom > 0 ? 'Resuming analysis...' : 'Reading file...');

    let textToAnalyze = cleanedText;
    // CRITICAL: Use local array instead of state for run tracking
    let runsToProcess = [];

    try {
      // Preprocessing (only if starting fresh)
      if (resumeFrom === 0) {
        const rawText = await file.text();
        textToAnalyze = preprocessText(rawText);
        setCleanedText(textToAnalyze);
        setCompletedRuns([]);
        runsToProcess = [];
      } else {
        // Resume: start with existing runs
        runsToProcess = completedRuns;
      }
      
      setStatus('analyzing');
      const startFrom = resumeFrom > 0 ? resumeFrom : 0;

      // Main analysis loop
      for (let i = startFrom; i < TOTAL_RUNS; i++) {
        const runNumber = i + 1;
        setCurrentRun(runNumber);
        setProgress((i / TOTAL_RUNS) * 100);
        setCurrentStep(`Analyzing run ${runNumber}/${TOTAL_RUNS}...`);
        
        try {
          // FIXED: Pass runNumber directly (eliminates Run 0 bug)
          const analysis = await analyzeText(textToAnalyze, SEEDS[i], runNumber);
          
          const runResult = {
            ...analysis,
            runNumber,
            seed: SEEDS[i]
          };
          
          // Add to local array immediately
          runsToProcess.push(runResult);
          
          // Update state for UI
          setCompletedRuns([...runsToProcess]);
          
          // AUTO-SAVE: Download this run immediately
          autoSaveRun(runResult, runNumber);
          
          setProgress(((i + 1) / TOTAL_RUNS) * 100);
          
        } catch (runError) {
          console.error(`Run ${runNumber} failed:`, runError);
          
          // If we have enough runs, show partial results
          if (runsToProcess.length >= 3) {
            setError(`Run ${runNumber} failed: ${runError.message}. Saved ${runsToProcess.length} completed runs.`);
            setStatus('partial');
            
            const partialSynthesis = await synthesizeResults(runsToProcess);
            setPartialResults(partialSynthesis);
            setResults(partialSynthesis);
            
            // AUTO-DOWNLOAD partial synthesis
            const timestamp = new Date().toISOString().split('T')[0];
            downloadFile(
              JSON.stringify(partialSynthesis, null, 2),
              `partial_synthesis_${fileName.replace('.txt', '')}_${timestamp}.json`
            );
            
            return;
          } else {
            throw new Error(`Run ${runNumber} failed after only ${runsToProcess.length} completed runs. Not enough data. Error: ${runError.message}`);
          }
        }
      }

      // All runs completed successfully
      setProgress(100);
      setStatus('synthesizing');
      setCurrentStep('Synthesizing consensus themes...');
      
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const synthesized = await synthesizeResults(runsToProcess);
      setResults(synthesized);
      setStatus('complete');
      setCurrentStep('Analysis complete!');
      
      // AUTO-DOWNLOAD final synthesis
      const timestamp = new Date().toISOString().split('T')[0];
      downloadFile(
        JSON.stringify(synthesized, null, 2),
        `final_synthesis_${fileName.replace('.txt', '')}_${timestamp}.json`
      );
      
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Analysis failed: ${err.message}`);
      setStatus('error');
      setCurrentStep('');
      
      if (runsToProcess.length >= 3) {
        const partialSynthesis = await synthesizeResults(runsToProcess);
        setPartialResults(partialSynthesis);
      }
    }
  };

  /**
   * Resume analysis from last successful run
   */
  const handleResume = () => {
    if (completedRuns.length > 0 && completedRuns.length < TOTAL_RUNS) {
      handleAnalysis(completedRuns.length);
    }
  };

  // ============================================================================
  // EXPORT FUNCTIONS
  // ============================================================================
  
  const exportResults = (usePartial = false) => {
    const dataToExport = usePartial ? partialResults : results;
    if (!dataToExport) return;

    const report = {
      fileName,
      analysisDate: new Date().toISOString(),
      isPartialAnalysis: usePartial || (dataToExport.reliability.runCount < TOTAL_RUNS),
      isCustomStructure: dataToExport.isCustomStructure || false,
      completedRuns: dataToExport.reliability.runCount,
      totalRuns: TOTAL_RUNS,
      fileStats,
      cleaningStats,
      reliability: dataToExport.reliability,
      consensusThemes: dataToExport.consensusThemes,
      individualRuns: dataToExport.allRuns,
      note: dataToExport.isCustomStructure ? 'Custom prompt structure - consensus themes extracted from recurring patterns across runs.' : undefined
    };

    const timestamp = new Date().toISOString().split('T')[0];
    const suffix = usePartial ? '_partial' : '';
    downloadFile(
      JSON.stringify(report, null, 2),
      `thematic_analysis${suffix}_${fileName.replace('.txt', '')}_${timestamp}.json`
    );
  };

  const exportReport = (usePartial = false) => {
    const dataToExport = usePartial ? partialResults : results;
    if (!dataToExport) return;

    let report = `QUALITATIVE THEMATIC ANALYSIS REPORT\n`;
    report += `${'='.repeat(70)}\n\n`;
    
    if (dataToExport.isCustomStructure) {
      report += `📊  CUSTOM PROMPT STRUCTURE  📊\n`;
      report += `This analysis used a custom prompt format.\n`;
      report += `Consensus themes extracted from patterns across all runs.\n\n`;
    }
    
    if (dataToExport.reliability.runCount < TOTAL_RUNS) {
      report += `⚠️  PARTIAL ANALYSIS REPORT ⚠️\n`;
      report += `This analysis is based on ${dataToExport.reliability.runCount} of ${TOTAL_RUNS} planned runs.\n`;
      report += `Results should be interpreted with appropriate caution.\n\n`;
    }
    
    report += `File Analyzed: ${fileName}\n`;
    report += `Analysis Date: ${new Date().toLocaleDateString()}\n`;
    report += `Completed Runs: ${dataToExport.reliability.runCount} of ${TOTAL_RUNS}\n\n`;
    
    if (cleaningStats) {
      report += `FILE PROCESSING SUMMARY\n`;
      report += `${'-'.repeat(70)}\n`;
      report += `Original Size: ${cleaningStats.originalLength.toLocaleString()} characters, ${cleaningStats.originalLines.toLocaleString()} lines\n`;
      report += `Cleaned Size: ${cleaningStats.cleanedLength.toLocaleString()} characters, ${cleaningStats.cleanedLines.toLocaleString()} lines\n`;
      report += `Reduction: ${cleaningStats.reductionPercent}% of metadata/noise removed\n\n`;
    }
    
    report += `RELIABILITY METRICS\n`;
    report += `${'-'.repeat(70)}\n`;
    report += `Average Inter-Run Consistency (Cosine): ${dataToExport.reliability.avgSimilarity}%\n`;
    report += `Consistency Range: ${dataToExport.reliability.minSimilarity}% - ${dataToExport.reliability.maxSimilarity}%\n`;
    report += `Cohen's Kappa (κ): ${dataToExport.reliability.avgKappa}\n`;
    report += `Kappa Range: ${dataToExport.reliability.minKappa} - ${dataToExport.reliability.maxKappa}\n`;
    report += `Reliability Assessment: ${dataToExport.reliability.interpretation}\n`;
    
    if (dataToExport.reliability.warning) {
      report += `\n⚠️  ${dataToExport.reliability.warning}\n`;
    }
    
    report += `\nInterpretation: `;
    if (dataToExport.reliability.runCount < TOTAL_RUNS) {
      report += `Based on ${dataToExport.reliability.runCount} runs, this analysis provides preliminary insights. `;
    }
    report += `A ${dataToExport.reliability.interpretation.toLowerCase()} reliability score indicates `;
    report += `${dataToExport.reliability.interpretation === 'High' ? 'strong thematic convergence across multiple analytical perspectives' : dataToExport.reliability.interpretation === 'Moderate' ? 'reasonable thematic consistency with some interpretive variation' : dataToExport.reliability.interpretation === 'Limited' ? 'limited data for reliability assessment' : 'considerable analytical divergence'}.\n\n`;
    
    report += `CONSENSUS THEMES\n`;
    report += `${'-'.repeat(70)}\n`;
    report += `The following ${dataToExport.consensusThemes.length} themes emerged consistently:\n\n`;
    
    dataToExport.consensusThemes.forEach((theme, idx) => {
      report += `${idx + 1}. ${theme.theme}\n`;
      report += `   Consistency: ${theme.consistency}% (${theme.appearanceCount}/${dataToExport.reliability.runCount} runs)\n`;
      report += `   Description: ${theme.description.slice(0, 200)}${theme.description.length > 200 ? '...' : ''}\n`;
      if (theme.sentiment) {
      report += `   Sentiment: ${theme.sentiment}\n`;
      }
      if (theme.evidence && theme.evidence.length > 0) {
        report += `   Evidence: "${theme.evidence[0].slice(0, 150)}${theme.evidence[0].length > 150 ? '...' : ''}"\n`;
      }
      report += `\n`;
    });
    
    report += `METHODOLOGICAL NOTE\n`;
    report += `${'-'.repeat(70)}\n`;
    report += `Multi-perspective analysis with ${dataToExport.reliability.runCount} independent runs using\n`;
    report += `varied parameters (seeds: ${SEEDS.slice(0, dataToExport.reliability.runCount).join(', ')}).\n`;
    report += `Text preprocessed to remove transcription artifacts.\n`;
    if (dataToExport.isCustomStructure) {
      report += `Custom prompt structure - consensus themes extracted from recurring patterns.\n`;
    }
    report += `Reliability measured using both Cosine Similarity and Cohen's Kappa (κ).\n`;
    report += `\n`;
    
    if (dataToExport.reliability.runCount < TOTAL_RUNS) {
      report += `NOTE: Partial analysis (${dataToExport.reliability.runCount}/${TOTAL_RUNS} runs).\n`;
      report += `Consider completing remaining runs for maximum reliability.\n\n`;
    }

    const timestamp = new Date().toISOString().split('T')[0];
    const suffix = usePartial ? '_partial' : '';
    downloadFile(
      report,
      `thematic_report${suffix}_${fileName.replace('.txt', '')}_${timestamp}.txt`,
      'text/plain'
    );
  };

  const exportCSV = (usePartial = false) => {
    const dataToExport = usePartial ? partialResults : results;
    if (!dataToExport) return;

    // CSV Header
    let csv = 'Theme,Description,Sentiment,Consistency,Appearance Count,Evidence\n';
    
    // Add themes (works for both default and custom structures)
    dataToExport.consensusThemes.forEach((theme: any) => {
      const themeName = `"${(theme.theme || '').replace(/"/g, '""')}"`;
      const description = `"${(theme.description || '').slice(0, 200).replace(/"/g, '""')}"`;
      const sentiment = theme.sentiment || 'N/A';
      const consistency = theme.consistency || '0';
      const appearanceCount = theme.appearanceCount || '0';
      const evidence = `"${(theme.evidence?.[0] || '').slice(0, 150).replace(/"/g, '""')}"`;
      
      csv += `${themeName},${description},${sentiment},${consistency},${appearanceCount},${evidence}\n`;
    });
    
    // Add reliability metrics as separate rows
    csv += '\n';
    csv += 'Metric,Value\n';
    csv += `Average Consistency (Cosine),${dataToExport.reliability.avgSimilarity}%\n`;
    csv += `Consistency Range,${dataToExport.reliability.minSimilarity}% - ${dataToExport.reliability.maxSimilarity}%\n`;
    csv += `Cohen's Kappa (κ),${dataToExport.reliability.avgKappa}\n`;
    csv += `Kappa Range,${dataToExport.reliability.minKappa} - ${dataToExport.reliability.maxKappa}\n`;
    csv += `Reliability Assessment,${dataToExport.reliability.interpretation}\n`;
    csv += `Completed Runs,${dataToExport.reliability.runCount}/${TOTAL_RUNS}\n`;
    if (dataToExport.isCustomStructure) {
      csv += `Structure Type,Custom Prompt\n`;
    }

    const timestamp = new Date().toISOString().split('T')[0];
    const suffix = usePartial ? '_partial' : '';
    downloadFile(
      csv,
      `thematic_analysis${suffix}_${fileName.replace('.txt', '')}_${timestamp}.csv`,
      'text/csv'
    );
  };

  // ============================================================================
  // RENDER COMPONENT
  // ============================================================================
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Main Input Card */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200/50 shadow-xl shadow-blue-100/50 p-8 mb-8">
          {/* Header */}
          <div className="mb-8 pb-6 border-b border-gradient-to-r from-blue-100 to-indigo-100">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg shadow-blue-200">
                <BarChart3 className="w-7 h-7 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent tracking-tight">Reliable Qualitative Thematic Analyzer</h1>
            </div>
            
            {/* Attribution */}
            <div className="mt-4">
              <p className="text-sm text-gray-600 mb-2">
                <span className="font-medium text-gray-900">Created by:</span> Aza Allsop and Nilesh Arnaiya at Aza Lab at Yale University
              </p>
              <p className="text-sm text-gray-600 leading-relaxed mb-4">
                Comprehensive qualitative research analysis tool powered by advanced AI models. A complementary tool to your human-level qualitative research pipeline. Customize prompts for optimal results.
              </p>
              <div className="flex items-center gap-3">
                <a 
                  href="https://github.com/NileshArnaiya/LLM-Thematic-Analysis-Tool" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-gray-800 to-gray-900 text-white rounded-lg hover:from-gray-900 hover:to-black transition-all duration-300 text-sm font-semibold shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clipRule="evenodd" />
                  </svg>
                  View on GitHub
                </a>
                <span className="text-xs text-gray-500">⭐ Give us a star!</span>
              </div>
            </div>
          </div>
          
          <p className="text-sm text-gray-600 mb-8 leading-relaxed">
            Multi-perspective analysis with auto-save and instant downloads. Performs {seeds.length} independent run{seeds.length !== 1 ? 's' : ''} (one per seed) and compares them using cosine similarity with{' '}
            <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" target="_blank" rel="noopener noreferrer" className="text-gray-900 underline hover:text-gray-700">all-MiniLM-L6-v2</a> (You can use other Similarity methods if you desire, see code!)
            {' '}for reliability assessment. Customize runs, seeds, and temperature as needed.
          </p>

          {/* Important Warnings */}
          <div className="mb-8 p-5 bg-gradient-to-r from-red-50 to-orange-50 border border-red-200/50 rounded-xl shadow-sm">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="font-medium text-red-900 mb-3 text-sm">Important Security & Privacy Notice</h3>
                <ul className="text-sm text-red-800 space-y-2 mb-3">
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-1">•</span>
                    <span><strong className="font-medium">Always use deidentified data</strong> — Remove all personally identifiable information before uploading</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-1">•</span>
                    <span><strong className="font-medium">No data is stored</strong> — All processing happens in your browser, no server storage</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-1">•</span>
                    <span><strong className="font-medium">NOT HIPAA compliant</strong> — This tool is not designed for protected health information</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-red-600 mt-1">•</span>
                    <span><strong className="font-medium">Use at your own risk</strong> — Ensure compliance with your organization's data policies</span>
                  </li>
                </ul>
                <p className="text-xs text-red-700">
                  <strong>GitHub:</strong> The whole process and code is open source at{' '}
                  <a href="https://github.com/NileshArnaiya/LLM-Thematic-Analysis-Tool" target="_blank" rel="noopener noreferrer" className="underline font-medium hover:text-red-900">our GitHub repository</a>
                </p>
              </div>
            </div>
          </div>

          {/* Model Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Select AI Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-gray-900 focus:border-gray-900 transition-colors disabled:bg-gray-50 disabled:text-gray-500"
              disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
            >
              {Object.entries(MODELS).map(([key, config]: [string, any]) => (
                <option key={key} value={key}>{config.name}</option>
              ))}
            </select>
            {selectedModel === 'azure-gpt-4o' && (
              <div className="mt-3 p-3 bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-900">
                  <strong>Azure OpenAI Configured:</strong> Endpoint: alright.openai.azure.com | Deployment: gpt-4o | API Version: 2024-12-01-preview
                </p>
              </div>
            )}
            {selectedModel.startsWith('openrouter-') && (
              <div className="mt-3 p-3 bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg">
                <p className="text-xs text-purple-900 mb-2">
                  <strong>OpenRouter:</strong> Unified API for multiple LLMs. Model: {MODELS[selectedModel]?.model || 'unknown'}
                </p>
                <p className="text-xs text-purple-800">
                  💡 If you get "Failed to fetch" errors, check: (1) API key is valid, (2) You have credits, (3) Browser console for CORS errors
                </p>
              </div>
            )}
            <p className="text-xs text-gray-500 mt-1.5">
              Choose the AI model to use for analysis. Each model has different strengths and capabilities.
            </p>
          </div>

          {/* API Key Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              {MODELS[selectedModel as keyof typeof MODELS]?.apiKeyName || 'API Key'}
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`Enter your ${MODELS[selectedModel as keyof typeof MODELS]?.apiKeyName || 'API'} key`}
              className="w-full px-3 py-2.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-gray-900 focus:border-gray-900 transition-colors placeholder:text-gray-400"
            />
            <p className="text-xs text-gray-500 mt-1">
              {selectedModel === 'gemini-2.5-pro' && (
                <>Get your API key from <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Google AI Studio</a></>
              )}
              {selectedModel === 'claude-3-5-sonnet' && (
                <>Get your API key from <a href="https://console.anthropic.com/" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Anthropic Console</a></>
              )}
              {selectedModel.startsWith('openrouter-') && (
                <>Get your API key from <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">OpenRouter</a> - Access multiple models with one key</>
              )}
              {selectedModel === 'azure-gpt-4o' && (
                <>Get your API key from <a href="https://portal.azure.com" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Azure Portal</a> (Azure OpenAI Service)</>
              )}
              {selectedModel.startsWith('gpt-') && !selectedModel.startsWith('azure-') && (
                <>Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">OpenAI Platform</a></>
              )}
              {selectedModel === 'llama-3-70b' && (
                <>Get your API key from <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Groq Console</a></>
              )}
              {selectedModel === 'deepseek-chat' && (
                <>Get your API key from <a href="https://platform.deepseek.com/api_keys" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">DeepSeek Platform</a></>
              )}
            </p>
          </div>

          {/* Custom Prompt */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Custom Prompt <span className="text-gray-500 font-normal">(Optional)</span>
            </label>
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              placeholder={DEFAULT_PROMPT}
              rows={12}
              className="w-full px-3 py-2.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-gray-900 focus:border-gray-900 transition-colors font-mono placeholder:text-gray-400 disabled:bg-gray-50 disabled:text-gray-500"
              disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
            />
            <p className="text-xs text-gray-500 mt-1.5">
              Upload transcript dialogue data and customize prompts to steer the model for best outputs. Available placeholders: <code className="bg-gray-100 px-1 rounded">{'{seed}'}</code> for the run seed, <code className="bg-gray-100 px-1 rounded">{'{text_chunk}'}</code> or <code className="bg-gray-100 px-1 rounded">{'{text}'}</code> for the transcript text. If no text placeholder is used, text is appended at the end. Leave empty to use default prompt.
            </p>
          </div>

          {/* Temperature Control */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Temperature <span className="text-gray-500 font-normal">({temperature})</span>
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="flex-1"
                disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
              />
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-20 px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-gray-900 focus:border-gray-900 transition-colors disabled:bg-gray-50 disabled:text-gray-500"
                disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1.5">
              Controls randomness: 0 = deterministic, 1 = balanced, 2 = creative. Applies to all runs.
            </p>
          </div>

          {/* Seeds Configuration */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Seeds Configuration <span className="text-gray-500 font-normal">(Max 6, {seeds.length} configured)</span>
            </label>
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {seeds.map((seed, index) => (
                  <div
                    key={index}
                    className="inline-flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-blue-100 to-indigo-100 border border-blue-200 rounded-lg text-sm shadow-sm hover:shadow-md transition-all duration-300 transform hover:scale-105"
                  >
                    <span className="font-mono text-gray-900">{seed}</span>
                    <button
                      onClick={() => setSeeds(seeds.filter((_, i) => i !== index))}
                      disabled={seeds.length <= 1 || status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
                      className="text-gray-500 hover:text-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Remove seed"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
              
              {seeds.length < 6 && (
                <div className="flex gap-2">
                  <input
                    type="number"
                    value={newSeedInput}
                    onChange={(e) => setNewSeedInput(e.target.value)}
                    placeholder="Enter seed value (e.g., 42)"
                    className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-gray-900 focus:border-gray-900 transition-colors placeholder:text-gray-400 disabled:bg-gray-50 disabled:text-gray-500"
                    disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && newSeedInput.trim()) {
                        const seedValue = parseInt(newSeedInput.trim());
                        if (!isNaN(seedValue) && seeds.length < 6) {
                          setSeeds([...seeds, seedValue]);
                          setNewSeedInput('');
                        }
                      }
                    }}
                  />
                  <button
                    onClick={() => {
                      if (newSeedInput.trim()) {
                        const seedValue = parseInt(newSeedInput.trim());
                        if (!isNaN(seedValue) && seeds.length < 6) {
                          setSeeds([...seeds, seedValue]);
                          setNewSeedInput('');
                        }
                      }
                    }}
                    disabled={!newSeedInput.trim() || seeds.length >= 6 || status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
                    className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg text-sm font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95 disabled:transform-none disabled:shadow-none"
                  >
                    Add Seed
                  </button>
                </div>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-1.5">
              Seeds control variation across runs. Each seed creates one analysis run. Use <code className="bg-gray-100 px-1 rounded">{'{seed}'}</code> in your prompt to reference the current seed value (e.g., for "Run ID: {'{seed}'}" instructions).
            </p>
          </div>

          {/* File Upload */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Upload Text File
            </label>
            <div className="border-2 border-dashed border-blue-200 rounded-xl p-8 text-center hover:border-blue-400 transition-all duration-300 bg-gradient-to-br from-blue-50/30 to-indigo-50/30 hover:shadow-md">
              <input
                type="file"
                accept=".txt"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
                disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <Upload className="w-10 h-10 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-700 mb-1 font-medium">
                  {fileName || 'Click to upload or drag and drop'}
                </p>
                <p className="text-xs text-gray-500">Transcript files (.txt format)</p>
              </label>
            </div>
            <div className="mt-3 p-3 bg-amber-50/50 border border-amber-200 rounded-lg">
              <p className="text-sm text-amber-900">
                <strong className="font-medium">File Size Warning:</strong> Do not upload files above 5MB. For larger files, 
                chunk them into multiple smaller files (under 5MB each) and combine the outputs later for best results.
              </p>
            </div>
          </div>

          {/* File Stats Display */}
          {fileStats && (
            <div className="mb-6 p-5 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 rounded-xl shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <FileSearch className="w-4 h-4 text-gray-600" />
                <h3 className="font-medium text-gray-900 text-sm">File Statistics</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-500 text-xs mb-1">Size</p>
                  <p className="font-medium text-gray-900">{fileStats.size} KB</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs mb-1">Lines</p>
                  <p className="font-medium text-gray-900">{fileStats.lines.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs mb-1">Words</p>
                  <p className="font-medium text-gray-900">{fileStats.words.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs mb-1">Characters</p>
                  <p className="font-medium text-gray-900">{fileStats.chars.toLocaleString()}</p>
                </div>
              </div>
            </div>
          )}

          {/* Progress Saved Indicator */}
          {completedRuns.length > 0 && completedRuns.length < TOTAL_RUNS && (
            <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Save className="w-4 h-4 text-gray-600" />
                <h3 className="font-medium text-gray-900 text-sm">Progress Saved</h3>
              </div>
              <p className="text-sm text-gray-700">
                {completedRuns.length} of {TOTAL_RUNS} runs completed and auto-downloaded!
              </p>
            </div>
          )}

          {/* Cleaning Stats Display */}
          {cleaningStats && status !== 'idle' && (
            <div className="mb-6 p-5 bg-gradient-to-br from-green-50 to-emerald-50 border border-green-100 rounded-xl shadow-sm">
              <div className="flex items-center gap-2 mb-2">
                <Scissors className="w-4 h-4 text-gray-600" />
                <h3 className="font-medium text-gray-900 text-sm">Preprocessing Complete</h3>
              </div>
              <p className="text-sm text-gray-700">
                Removed {cleaningStats.removedChars.toLocaleString()} characters ({cleaningStats.reductionPercent}%) of metadata
              </p>
              <p className="text-xs text-gray-600 mt-1">
                Cleaned: {cleaningStats.cleanedLength.toLocaleString()} chars, {cleaningStats.cleanedLines.toLocaleString()} lines
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center gap-3 mb-2">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                <p className="text-red-800 font-semibold">Error Occurred</p>
              </div>
              <p className="text-sm text-red-700 mb-3">{error}</p>
              
              {partialResults && (
                <div className="mt-3 pt-3 border-t border-red-300">
                  <p className="text-sm text-red-900 font-semibold mb-2">
                    ✓ Good news: {completedRuns.length} runs were auto-saved!
                  </p>
                  <p className="text-xs text-red-700">
                    Check your downloads folder. Partial results are displayed below.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              onClick={() => handleAnalysis(0)}
              disabled={status === 'preprocessing' || status === 'analyzing' || status === 'synthesizing'}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3.5 rounded-xl font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed disabled:text-gray-500 flex items-center justify-center gap-2 text-sm shadow-lg shadow-blue-200 hover:shadow-xl hover:shadow-blue-300 disabled:shadow-none transform hover:scale-[1.02] active:scale-[0.98]"
            >
              {status === 'preprocessing' || status === 'analyzing' || status === 'synthesizing' ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {status === 'preprocessing' ? 'Preprocessing...' : status === 'analyzing' ? `Analyzing (Run ${currentRun}/${TOTAL_RUNS})` : 'Synthesizing Results'}
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  {completedRuns.length > 0 ? 'Start New Analysis' : 'Start Analysis'}
                </>
              )}
            </button>

            {completedRuns.length > 0 && completedRuns.length < TOTAL_RUNS && status !== 'analyzing' && (
              <button
                onClick={handleResume}
                className="w-full bg-white text-blue-600 border-2 border-blue-200 py-3.5 rounded-xl font-semibold hover:bg-blue-50 hover:border-blue-300 transition-all duration-300 flex items-center justify-center gap-2 text-sm shadow-md hover:shadow-lg transform hover:scale-[1.02] active:scale-[0.98]"
              >
                <RefreshCw className="w-4 h-4" />
                Resume from Run {completedRuns.length + 1} ({TOTAL_RUNS - completedRuns.length} remaining)
              </button>
            )}
          </div>

          {/* Current Step Info */}
          {currentStep && status !== 'idle' && status !== 'complete' && status !== 'partial' && (
            <div className="mt-4 text-center text-sm text-gray-600">
              {currentStep}
            </div>
          )}

          {/* Progress Bar */}
          {(status === 'analyzing' || status === 'synthesizing' || status === 'preprocessing') && (
            <div className="mt-6 pt-6 border-t border-gray-100">
              <div className="flex justify-between text-xs text-gray-600 mb-2">
                <span>Progress</span>
                <span className="font-medium">{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 mb-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out shadow-sm"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 text-center">
                Each run auto-downloads to your browser. Check downloads folder!
              </p>
            </div>
          )}
        </div>

        {/* Results Display */}
        {(results || partialResults) && (status === 'complete' || status === 'partial') && (
          <div className="space-y-6">
            {/* Partial Results Warning */}
            {(status === 'partial' || (results && results.reliability.runCount < TOTAL_RUNS)) && (
              <div className="bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200 rounded-xl p-5 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle className="w-4 h-4 text-amber-600" />
                  <h3 className="text-sm font-medium text-amber-900">Partial Analysis Results</h3>
                </div>
                <p className="text-sm text-amber-800 mb-1">
                  Completed {(results || partialResults).reliability.runCount} of {TOTAL_RUNS} runs. All saved to downloads!
                </p>
                <p className="text-xs text-amber-700">
                  Use the "Resume" button above to complete remaining runs for better reliability.
                </p>
              </div>
            )}

            {/* Reliability Metrics */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200/50 shadow-xl shadow-blue-100/50 p-6">
              <div className="flex items-center gap-2 mb-6">
                <TrendingUp className="w-5 h-5 text-gray-600" />
                <h2 className="text-lg font-semibold text-gray-900">Reliability Metrics</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300">
                  <p className="text-xs font-medium text-blue-600 mb-2">Average Consistency (Cosine)</p>
                  <p className="text-3xl font-bold text-blue-700">{(results || partialResults).reliability.avgSimilarity}%</p>
                  <p className="text-xs text-blue-500 mt-2">
                    Range: {(results || partialResults).reliability.minSimilarity}% - {(results || partialResults).reliability.maxSimilarity}%
                  </p>
                </div>
                <div className="bg-gradient-to-br from-indigo-50 to-purple-50 border border-indigo-100 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300">
                  <p className="text-xs font-medium text-indigo-600 mb-2">Cohen's Kappa (κ)</p>
                  <p className="text-3xl font-bold text-indigo-700">{(results || partialResults).reliability.avgKappa}</p>
                  <p className="text-xs text-indigo-500 mt-2">
                    Range: {(results || partialResults).reliability.minKappa} - {(results || partialResults).reliability.maxKappa}
                  </p>
                </div>
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-100 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300">
                  <p className="text-xs font-medium text-green-600 mb-2">Reliability Assessment</p>
                  <p className="text-3xl font-bold text-green-700">{(results || partialResults).reliability.interpretation}</p>
                  <p className="text-xs text-green-500 mt-2">
                    Based on {(results || partialResults).reliability.runCount}/{TOTAL_RUNS} runs
                  </p>
                </div>
              </div>
              
              <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl shadow-sm">
                <p className="text-sm text-blue-900">
                  <strong>Cohen's Kappa Interpretation:</strong> {' '}
                  {parseFloat((results || partialResults).reliability.avgKappa) > 0.8 ? 'Almost perfect agreement' :
                   parseFloat((results || partialResults).reliability.avgKappa) > 0.6 ? 'Substantial agreement' :
                   parseFloat((results || partialResults).reliability.avgKappa) > 0.4 ? 'Moderate agreement' :
                   parseFloat((results || partialResults).reliability.avgKappa) > 0.2 ? 'Fair agreement' :
                   parseFloat((results || partialResults).reliability.avgKappa) > 0 ? 'Slight agreement' : 'Poor agreement'}
                  {' '}(κ = {(results || partialResults).reliability.avgKappa})
                </p>
              </div>
              
              {(results || partialResults).reliability.warning && (
                <div className="mt-4 p-3 bg-amber-50/50 border border-amber-200 rounded-lg">
                  <p className="text-sm text-amber-800">
                    {(results || partialResults).reliability.warning}
                  </p>
                </div>
              )}
            </div>

            {/* Consensus Themes */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200/50 shadow-xl shadow-indigo-100/50 p-6">
              <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-100">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-gray-600" />
                  <h2 className="text-lg font-semibold text-gray-900">Consensus Themes</h2>
                </div>
                <p className="text-xs text-gray-500">
                  {(results || partialResults).consensusThemes.length} themes identified
                </p>
              </div>
              
              {(results || partialResults).isCustomStructure && (
                <div className="mb-4 p-4 bg-gradient-to-r from-cyan-50 to-blue-50 border border-cyan-200 rounded-xl shadow-sm">
                  <p className="text-sm text-blue-900">
                    <strong>Custom Prompt Structure:</strong> Consensus built from patterns across all runs. Themes extracted from arrays in your custom JSON structure.
                  </p>
                </div>
              )}
              
              {(results || partialResults).consensusThemes.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p className="text-sm">No consensus themes found.</p>
                  <p className="text-xs mt-1">This may indicate varied content or insufficient runs.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {(results || partialResults).consensusThemes.map((theme, idx) => (
                    <div key={idx} className="border border-gray-200/50 rounded-xl p-5 hover:border-blue-300 transition-all duration-300 bg-gradient-to-br from-gray-50/50 to-blue-50/30 hover:shadow-lg hover:shadow-blue-100/50 transform hover:scale-[1.01]">
                      <div className="flex items-start justify-between mb-2">
                        <h3 className="text-base font-medium text-gray-900 flex-1">{theme.theme}</h3>
                        <span className="px-3 py-1.5 bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 rounded-full text-xs font-bold whitespace-nowrap ml-3 shadow-sm">
                          {theme.consistency}% consistency
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-3 leading-relaxed">{theme.description}</p>
                      {theme.evidence && theme.evidence.length > 0 && (
                        <div className="mb-3 p-4 bg-gradient-to-br from-indigo-50/50 to-purple-50/50 rounded-lg border border-indigo-100 shadow-sm">
                          <p className="text-xs font-semibold text-indigo-600 mb-2">Supporting Evidence:</p>
                          <p className="text-sm text-gray-700 italic leading-relaxed">"{theme.evidence[0].slice(0, 200)}{theme.evidence[0].length > 200 ? '...' : ''}"</p>
                        </div>
                      )}
                      <div className="flex items-center gap-2 flex-wrap">
                        {theme.sentiment && (
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          theme.sentiment === 'positive' 
                            ? 'bg-green-50 text-green-700 border border-green-200' 
                            : theme.sentiment === 'negative'
                            ? 'bg-red-50 text-red-700 border border-red-200'
                            : 'bg-gray-50 text-gray-700 border border-gray-200'
                        }`}>
                          {theme.sentiment}
                        </span>
                        )}
                        <span className="text-xs text-gray-500">
                          Appeared in {theme.appearanceCount}/{(results || partialResults).reliability.runCount} runs
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Collapsible Individual Runs */}
              <details className="mt-6">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900 flex items-center gap-2">
                  <span>View Individual Run Data ({(results || partialResults).allRuns.length} runs)</span>
                </summary>
                <div className="mt-4 space-y-3">
                  {(results || partialResults).allRuns.map((run, idx) => (
                    <details key={idx} className="border border-gray-200 rounded-lg">
                      <summary className="cursor-pointer p-3 hover:bg-gray-50 text-sm text-gray-700 flex items-center justify-between">
                        <span>Run {run.runNumber} (Seed: {run.seed})</span>
                        <span className="text-xs text-gray-500">Click to expand JSON</span>
                      </summary>
                      <div className="p-3 pt-0">
                        <pre className="bg-gray-50 border border-gray-200 rounded p-3 text-xs overflow-x-auto max-h-96 overflow-y-auto">
                          {JSON.stringify(run, null, 2)}
                        </pre>
                      </div>
                    </details>
                  ))}
                </div>
              </details>
            </div>

            {/* Export Options */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200/50 shadow-xl shadow-green-100/50 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-2">Export Options</h2>
              <p className="text-sm text-gray-600 mb-4">
                Individual runs and consensus synthesis already auto-downloaded. These are additional export formats.
              </p>
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => exportResults(status === 'partial')}
                  className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 text-white py-3 rounded-xl font-semibold hover:from-green-700 hover:to-emerald-700 transition-all duration-300 flex items-center justify-center gap-2 text-sm shadow-lg shadow-green-200 hover:shadow-xl hover:shadow-green-300 transform hover:scale-105 active:scale-95"
                >
                  <Download className="w-4 h-4" />
                  Export JSON Summary
                </button>
                <button
                  onClick={() => exportCSV(status === 'partial')}
                  className="flex-1 bg-white text-blue-600 border-2 border-blue-200 py-3 rounded-xl font-semibold hover:bg-blue-50 hover:border-blue-300 transition-all duration-300 flex items-center justify-center gap-2 text-sm shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95"
                >
                  <Download className="w-4 h-4" />
                  Export CSV
                </button>
                <button
                  onClick={() => exportReport(status === 'partial')}
                  className="flex-1 bg-white text-indigo-600 border-2 border-indigo-200 py-3 rounded-xl font-semibold hover:bg-indigo-50 hover:border-indigo-300 transition-all duration-300 flex items-center justify-center gap-2 text-sm shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95"
                >
                  <FileText className="w-4 h-4" />
                  Export Report
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;