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
  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  
  const MAX_RETRIES = 3;
  const RETRY_DELAY_BASE = 1500;
  const TOTAL_RUNS = 6;
  const SEEDS = [42, 123, 456, 789, 1011, 1213];

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
  // GEMINI API INTEGRATION
  // ============================================================================
  
  /**
   * Call Gemini API with comprehensive error handling
   * 
   * Key Features:
   * - Structured prompt for consistent JSON responses
   * - Temperature variation based on seed (introduces diversity)
   * - Robust JSON parsing with fallback strategies
   * - Detailed error categorization (auth, rate limit, network)
   * 
   * Model Choice: gemini-2.0-flash-exp
   * - Faster than Pro models (lower latency)
   * - Good quality for qualitative analysis
   * - Better rate limits
   * 
   * @param {string} text - Text to analyze
   * @param {number} seed - Seed for temperature variation
   * @param {boolean} isChunked - Is this part of a multi-chunk analysis?
   * @param {number} chunkIndex - Index of current chunk (0-based)
   * @param {number} totalChunks - Total number of chunks
   * @returns {Object} - Parsed analysis results
   */
  const callGeminiAPI = async (text: string, seed: number, isChunked: boolean = false, chunkIndex: number = 0, totalChunks: number = 1) => {
    const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${apiKey}`;
    
    const contextPrefix = isChunked 
      ? `This is chunk ${chunkIndex + 1} of ${totalChunks}. Analyze only this section.`
      : 'Analyze the complete text below.';
    
    const prompt = `${contextPrefix}

Conduct a qualitative thematic analysis. Identify major emotional, psychological, and conceptual themes.

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
}

Text: ${text}`;

    let response;
    try {
      response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            // Vary temperature by seed for diversity across runs
            temperature: 0.7 + (seed % 100) / 200,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 8192,
          }
        })
      });
    } catch (fetchError) {
      const errorMessage = fetchError instanceof Error ? fetchError.message : String(fetchError);
      throw new Error(`Network error: ${errorMessage}`);
    }

    // Handle HTTP errors with specific guidance
    if (!response.ok) {
      let errorMessage = response.statusText;
      
      try {
        const errorData = await response.json();
        errorMessage = errorData.error?.message || errorMessage;
        
        if (response.status === 429) {
          throw new Error(`Rate limit exceeded. Wait and retry. (${errorMessage})`);
        } else if (response.status === 401 || response.status === 403) {
          throw new Error(`Authentication failed. Check API key. (${errorMessage})`);
        } else if (response.status === 503) {
          throw new Error(`Service unavailable. Will retry. (${errorMessage})`);
        }
      } catch (jsonError) {
        // If error response isn't JSON, use status text
      }
      
      throw new Error(`API error (${response.status}): ${errorMessage}`);
    }

    const data = await response.json();
    const textResponse = data.candidates?.[0]?.content?.parts?.[0]?.text;
    
    if (!textResponse) {
      throw new Error('No response from API');
    }

    // Robust JSON parsing - handle markdown code blocks
    let jsonStr = textResponse.trim();
    
    if (jsonStr.startsWith('```json')) {
      jsonStr = jsonStr.replace(/```json\s*/g, '').replace(/```\s*$/g, '');
    } else if (jsonStr.startsWith('```')) {
      jsonStr = jsonStr.replace(/```\s*/g, '').replace(/```\s*$/g, '');
    }
    
    try {
      const parsed = JSON.parse(jsonStr);
      
      // Validate response structure
      if (!parsed.majorEmotionalThemes || !Array.isArray(parsed.majorEmotionalThemes)) {
        throw new Error('Invalid response structure');
      }
      if (!parsed.emotionalPatterns || typeof parsed.emotionalPatterns !== 'object') {
        throw new Error('Missing emotional patterns');
      }
      
      return parsed;
    } catch (parseError) {
      console.error('Parse error:', textResponse);
      const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);
      throw new Error(`Parse failed: ${errorMessage}`);
    }
  };

  // Wrap API call with retry logic
  const callGeminiAPIWithRetry = withRetry(callGeminiAPI);

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
      return await callGeminiAPIWithRetry(chunks[0], seed, false);
    }
    
    // Multiple chunks - analyze each and merge
    const chunkResults = [];
    for (let i = 0; i < chunks.length; i++) {
      setCurrentStep(`Run ${runNumber}: Analyzing chunk ${i + 1}/${chunks.length}`);
      const result = await callGeminiAPIWithRetry(chunks[i], seed, true, i, chunks.length);
      chunkResults.push(result);
    }
    
    setCurrentStep(`Run ${runNumber}: Merging chunk analyses`);
    return mergeChunkAnalyses(chunkResults);
  };

  /**
   * Intelligently merge results from multiple chunks
   * 
   * Strategy:
   * - Combine duplicate themes (by name)
   * - Sum frequencies for combined themes
   * - Deduplicate evidence quotes
   * - Average sentiment scores
   * - Find most common dominant tone
   * 
   * @param {Object[]} chunkResults - Array of analysis results from chunks
   * @returns {Object} - Merged analysis result
   */
  const mergeChunkAnalyses = (chunkResults) => {
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
      allInsights.push(...result.keyInsights);
      
      // Aggregate sentiment
      totalSentiment += result.emotionalPatterns.overallSentiment;
      
      // Count tone occurrences
      const tone = result.emotionalPatterns.dominantTone;
      toneFrequency[tone] = (toneFrequency[tone] || 0) + 1;
    });
    
    // Find most common tone
    const dominantTone = Object.entries(toneFrequency)
      .sort((a, b) => b[1] - a[1])[0][0];
    
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
  // If the embedding model isn't available, fall back to Jaccard similarity
  // This ensures your app still works even if the model fails to load
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

    // Generate embeddings for all themes in both runs
    // The model converts each text string into a 384-dimensional vector
    // that captures its semantic meaning
    const embeddings1 = await extractorRef.current(themes1, { 
      pooling: 'mean',  // Average across tokens in each theme
      normalize: true   // Normalize vectors to unit length (important for cosine similarity)
    });
    
    const embeddings2 = await extractorRef.current(themes2, { 
      pooling: 'mean', 
      normalize: true 
    });

    // Each run might have multiple themes, so we need to combine them into
    // a single representative vector. We do this by averaging all theme vectors.
    // This gives us one vector that represents the "overall semantic content"
    // of all themes in that run.
    
    // embeddings1.data is a flat array, so we need to reshape it
    const numThemes1 = themes1.length;
    const embeddingDim = embeddings1.data.length / numThemes1;
    
    const numThemes2 = themes2.length;
    
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

    // Aggregate theme data across all runs
    const themeFrequency = {};
    const themeDescriptions = {};
    const themeSentiments = {};
    const themeEvidence = {};

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

    // Calculate inter-run similarities
    let avgSimilarity = 0;
    let minSimilarity = 1;
    let maxSimilarity = 0;
    
    // NEW CODE:
if (allRuns.length > 1) {
    const similarities = [];
    // We need to calculate all pairwise similarities between runs
    // Now we await each calculation since it involves the embedding model
    for (let i = 0; i < allRuns.length; i++) {
      for (let j = i + 1; j < allRuns.length; j++) {
        const sim = await calculateSimilarity(allRuns[i], allRuns[j]);
        similarities.push(sim);
      }
    }
    avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    minSimilarity = Math.min(...similarities);
    maxSimilarity = Math.max(...similarities);
  }
    else {
      avgSimilarity = 1;
      minSimilarity = 1;
      maxSimilarity = 1;
    }

    // Adaptive consensus threshold (more lenient for fewer runs)
    const consensusThreshold = Math.max(1, Math.ceil(allRuns.length / 2));
    
    const consensusThemes = Object.entries(themeFrequency)
      .filter(([_, freq]) => freq >= consensusThreshold)
      .map(([theme, freq]) => ({
        theme,
        description: themeDescriptions[theme],
        sentiment: themeSentiments[theme],
        consistency: (freq / allRuns.length * 100).toFixed(1),
        appearanceCount: freq,
        evidence: [...new Set(themeEvidence[theme])].slice(0, 5)
      }))
      .sort((a, b) => parseFloat(b.consistency) - parseFloat(a.consistency));

    // Generate interpretation and warnings
    let interpretation;
    let warning = null;
    
    if (allRuns.length < 3) {
      interpretation = 'Limited';
      warning = `Only ${allRuns.length} run(s) completed. Results should be interpreted with caution.`;
    } else if (allRuns.length < 6) {
      interpretation = avgSimilarity > 0.6 ? 'Moderate' : 'Low';
      warning = `Partial analysis (${allRuns.length} of 6 runs completed). Consider re-running for more reliable results.`;
    } else {
      interpretation = avgSimilarity > 0.7 ? 'High' : avgSimilarity > 0.5 ? 'Moderate' : 'Low';
    }

    return {
      consensusThemes,
      reliability: {
        avgSimilarity: (avgSimilarity * 100).toFixed(1),
        minSimilarity: (minSimilarity * 100).toFixed(1),
        maxSimilarity: (maxSimilarity * 100).toFixed(1),
        interpretation,
        runCount: allRuns.length,
        warning
      },
      allRuns
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
      completedRuns: dataToExport.reliability.runCount,
      totalRuns: TOTAL_RUNS,
      fileStats,
      cleaningStats,
      reliability: dataToExport.reliability,
      consensusThemes: dataToExport.consensusThemes,
      individualRuns: dataToExport.allRuns
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
    report += `Average Inter-Run Consistency: ${dataToExport.reliability.avgSimilarity}%\n`;
    report += `Consistency Range: ${dataToExport.reliability.minSimilarity}% - ${dataToExport.reliability.maxSimilarity}%\n`;
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
      report += `   Description: ${theme.description}\n`;
      report += `   Sentiment: ${theme.sentiment}\n`;
      if (theme.evidence && theme.evidence.length > 0) {
        report += `   Evidence: "${theme.evidence[0]}"\n`;
      }
      report += `\n`;
    });
    
    report += `METHODOLOGICAL NOTE\n`;
    report += `${'-'.repeat(70)}\n`;
    report += `Multi-perspective analysis with ${dataToExport.reliability.runCount} independent runs using\n`;
    report += `varied parameters (seeds: ${SEEDS.slice(0, dataToExport.reliability.runCount).join(', ')}).\n`;
    report += `Text preprocessed to remove transcription artifacts.\n\n`;
    
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

  // ============================================================================
  // RENDER COMPONENT
  // ============================================================================
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Main Input Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          {/* Header */}
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">Reliable Qualitative Thematic Analyzer</h1>
          </div>
          
          {/* Attribution */}
          <div className="mb-6 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-lg">
            <p className="text-sm text-gray-700 mb-2">
              <span className="font-semibold">Created by:</span> Aza Allsop and Nilesh Arnaiya at Aza Lab
            </p>
            <p className="text-sm text-gray-600">
              Comprehensive qualitative research analysis tool powered by Gemini 2.5 Pro, a complimentary tool to your human-level qualitative research pipeline. Easy to use with any file, change the prompts according to your text files to get the best results.
            </p>
          </div>
          
          <p className="text-gray-600 mb-8">
            Multi-perspective analysis with auto-save and instant downloads (We do 6 runs and average them out and compare them to get the best results with cosine similarity with https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Feel free to change the number of runs and seeds to get the best results.
          </p>

          {/* API Key Input */}
          <div className="mb-6">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Gemini API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your Gemini API key"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">
              Get your API key from <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Google AI Studio</a>
            </p>
          </div>

          {/* File Upload */}
          <div className="mb-6">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Upload Text File
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-400 transition-colors">
              <input
                type="file"
                accept=".txt"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
                disabled={status === 'analyzing' || status === 'preprocessing' || status === 'synthesizing'}
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-600 mb-1">
                  {fileName || 'Click to upload or drag and drop'}
                </p>
                <p className="text-sm text-gray-400">Large transcript files supported (.txt format)</p>
              </label>
            </div>
          </div>

          {/* File Stats Display */}
          {fileStats && (
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <FileSearch className="w-5 h-5 text-blue-600" />
                <h3 className="font-semibold text-blue-900">File Statistics</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Size</p>
                  <p className="font-semibold text-gray-800">{fileStats.size} KB</p>
                </div>
                <div>
                  <p className="text-gray-600">Lines</p>
                  <p className="font-semibold text-gray-800">{fileStats.lines.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-600">Words</p>
                  <p className="font-semibold text-gray-800">{fileStats.words.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-600">Characters</p>
                  <p className="font-semibold text-gray-800">{fileStats.chars.toLocaleString()}</p>
                </div>
              </div>
            </div>
          )}

          {/* Progress Saved Indicator */}
          {completedRuns.length > 0 && completedRuns.length < TOTAL_RUNS && (
            <div className="mb-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Save className="w-5 h-5 text-purple-600" />
                <h3 className="font-semibold text-purple-900">Progress Saved</h3>
              </div>
              <p className="text-sm text-gray-700">
                {completedRuns.length} of {TOTAL_RUNS} runs completed and auto-downloaded!
              </p>
            </div>
          )}

          {/* Cleaning Stats Display */}
          {cleaningStats && status !== 'idle' && (
            <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Scissors className="w-5 h-5 text-green-600" />
                <h3 className="font-semibold text-green-900">Preprocessing Complete</h3>
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
              className="w-full bg-indigo-600 text-white py-4 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            >
              {status === 'preprocessing' || status === 'analyzing' || status === 'synthesizing' ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  {status === 'preprocessing' ? 'Preprocessing...' : status === 'analyzing' ? `Analyzing (Run ${currentRun}/${TOTAL_RUNS})` : 'Synthesizing Results'}
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  {completedRuns.length > 0 ? 'Start New Analysis' : 'Start Analysis'}
                </>
              )}
            </button>

            {completedRuns.length > 0 && completedRuns.length < TOTAL_RUNS && status !== 'analyzing' && (
              <button
                onClick={handleResume}
                className="w-full bg-green-600 text-white py-4 rounded-lg font-semibold hover:bg-green-700 transition-colors flex items-center justify-center gap-3"
              >
                <RefreshCw className="w-5 h-5" />
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
            <div className="mt-6">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-indigo-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-2 text-center">
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
              <div className="bg-yellow-50 border-2 border-yellow-300 rounded-2xl p-6">
                <div className="flex items-center gap-3 mb-3">
                  <AlertCircle className="w-6 h-6 text-yellow-600" />
                  <h3 className="text-lg font-bold text-yellow-900">Partial Analysis Results</h3>
                </div>
                <p className="text-yellow-800 mb-2">
                  Completed {(results || partialResults).reliability.runCount} of {TOTAL_RUNS} runs. All saved to downloads!
                </p>
                <p className="text-sm text-yellow-700">
                  Use the "Resume" button above to complete remaining runs for better reliability.
                </p>
              </div>
            )}

            {/* Reliability Metrics */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <TrendingUp className="w-6 h-6 text-green-600" />
                <h2 className="text-2xl font-bold text-gray-800">Reliability Metrics</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
                  <p className="text-sm text-gray-600 mb-2">Average Consistency</p>
                  <p className="text-3xl font-bold text-blue-700">{(results || partialResults).reliability.avgSimilarity}%</p>
                </div>
                <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
                  <p className="text-sm text-gray-600 mb-2">Consistency Range</p>
                  <p className="text-2xl font-bold text-green-700">
                    {(results || partialResults).reliability.minSimilarity}% - {(results || partialResults).reliability.maxSimilarity}%
                  </p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl">
                  <p className="text-sm text-gray-600 mb-2">Reliability</p>
                  <p className="text-3xl font-bold text-purple-700">{(results || partialResults).reliability.interpretation}</p>
                  <p className="text-xs text-gray-600 mt-1">
                    Based on {(results || partialResults).reliability.runCount}/{TOTAL_RUNS} runs
                  </p>
                </div>
              </div>
              
              {(results || partialResults).reliability.warning && (
                <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
                  <p className="text-sm text-orange-800">
                    ⚠️ {(results || partialResults).reliability.warning}
                  </p>
                </div>
              )}
            </div>

            {/* Consensus Themes */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-6 h-6 text-indigo-600" />
                  <h2 className="text-2xl font-bold text-gray-800">Consensus Themes</h2>
                </div>
                <p className="text-sm text-gray-500">
                  {(results || partialResults).consensusThemes.length} themes identified
                </p>
              </div>
              
              {(results || partialResults).consensusThemes.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p>No consensus themes found.</p>
                  <p className="text-sm mt-2">This may indicate varied content or insufficient runs.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {(results || partialResults).consensusThemes.map((theme, idx) => (
                    <div key={idx} className="border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-xl font-semibold text-gray-800 flex-1">{theme.theme}</h3>
                        <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium whitespace-nowrap ml-4">
                          {theme.consistency}% consistency
                        </span>
                      </div>
                      <p className="text-gray-600 mb-3">{theme.description}</p>
                      {theme.evidence && theme.evidence.length > 0 && (
                        <div className="mb-3 p-3 bg-gray-50 rounded-lg">
                          <p className="text-xs text-gray-500 mb-1">Supporting Evidence:</p>
                          <p className="text-sm text-gray-700 italic">"{theme.evidence[0]}"</p>
                        </div>
                      )}
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={`px-3 py-1 rounded-full text-sm ${
                          theme.sentiment === 'positive' 
                            ? 'bg-green-100 text-green-700' 
                            : theme.sentiment === 'negative'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-gray-100 text-gray-700'
                        }`}>
                          {theme.sentiment}
                        </span>
                        <span className="text-sm text-gray-600">
                          Appeared in {theme.appearanceCount}/{(results || partialResults).reliability.runCount} runs
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Export Options */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Manual Export Options</h2>
              <p className="text-sm text-gray-600 mb-4">
                Note: Individual runs and final synthesis already auto-downloaded. These are additional export formats.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={() => exportResults(status === 'partial')}
                  className="flex-1 bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Export Full JSON
                </button>
                <button
                  onClick={() => exportReport(status === 'partial')}
                  className="flex-1 bg-purple-600 text-white py-3 rounded-lg font-semibold hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
                >
                  <FileText className="w-5 h-5" />
                  Export Text Report
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