# Reliable Qualitative Thematic Analyzer

A comprehensive qualitative research analysis tool powered by Gemini 2.5 Pro, designed as a complimentary tool to human-level qualitative research pipelines. This application performs multi-perspective thematic analysis with automatic reliability assessment and instant data export capabilities.

## 🎯 Overview

Created by **Aza Allsop and Nilesh Arnaiya at Aza Lab at Yale**, this tool provides:

- **Multi-perspective analysis** with 6 independent runs using varied parameters
- **Automatic reliability assessment** using cosine similarity with semantic embeddings
- **Auto-save functionality** - each run downloads immediately to prevent data loss
- **Resume capability** - continue from interrupted analyses
- **Comprehensive export options** - JSON data and formatted text reports

## 🚀 Key Features

### Core Analysis Engine
- **Gemini 2.5 Pro Integration**: Uses Google's latest language model for qualitative analysis
- **Semantic Similarity**: Implements Hugging Face's `all-MiniLM-L6-v2` model for reliability scoring
- **Multi-Run Strategy**: Performs 6 independent analyses with different seeds for consistency validation
- **Intelligent Chunking**: Automatically splits large texts while preserving semantic boundaries

### Reliability & Quality Assurance
- **Inter-Run Consistency**: Calculates similarity scores between analysis runs
- **Consensus Theme Identification**: Identifies themes that appear across multiple runs
- **Adaptive Thresholds**: Adjusts consensus requirements based on completed runs
- **Graceful Degradation**: Works with partial results when some runs fail

### User Experience
- **Auto-Download**: Every completed run saves automatically to browser downloads
- **Progress Tracking**: Real-time progress indicators and status updates
- **Resume Functionality**: Continue interrupted analyses from the last successful run
- **Error Recovery**: Robust error handling with retry mechanisms

## 🛠 Technical Implementation

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  Text Processing │───▶│  Multi-Run      │
│   & Validation  │    │  & Chunking      │    │  Analysis       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Auto-Download │◀───│  Reliability     │◀───│  Semantic      │
│   & Export      │    │  Assessment      │    │  Similarity     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Text Preprocessing (`preprocessText`)
```typescript
const preprocessText = (text: string) => {
  // Removes transcription artifacts, timestamps, and metadata
  // Calculates cleaning statistics for transparency
  // Returns cleaned text optimized for analysis
}
```

**Features:**
- Removes Otter.ai and other transcription service attributions
- Filters timestamp lines (00:00 format)
- Eliminates empty or very short lines
- Normalizes whitespace and formatting
- Provides detailed cleaning statistics

#### 2. Intelligent Text Chunking (`chunkText`)
```typescript
const chunkText = (text: string, maxChunkSize: number = 30000) => {
  // Respects paragraph boundaries
  // Falls back to sentence boundaries for large paragraphs
  // Only splits mid-sentence as last resort
}
```

**Strategy:**
1. **Paragraph Priority**: Attempts to keep paragraphs intact
2. **Sentence Fallback**: Splits oversized paragraphs by sentences
3. **Character Limit**: Enforces 30KB chunks for API efficiency
4. **Semantic Preservation**: Maintains context and meaning

#### 3. Gemini API Integration (`callGeminiAPI`)
```typescript
const callGeminiAPI = async (
  text: string, 
  seed: number, 
  isChunked: boolean = false, 
  chunkIndex: number = 0, 
  totalChunks: number = 1
) => {
  // Structured prompts for consistent JSON responses
  // Temperature variation based on seed for diversity
  // Robust error handling and retry logic
}
```

**Model Configuration:**
- **Model**: `gemini-2.5-pro` (latest version)
- **Temperature**: `0.7 + (seed % 100) / 200` (varies by run)
- **Max Tokens**: 8192 for comprehensive responses
- **Top-K**: 40, Top-P: 0.95 for balanced creativity/consistency

#### 4. Semantic Similarity Engine (`calculateSimilarity`)
```typescript
const calculateSimilarity = async (run1: any, run2: any) => {
  // Uses Hugging Face all-MiniLM-L6-v2 for embeddings
  // Calculates cosine similarity between theme vectors
  // Falls back to Jaccard similarity if model unavailable
}
```

**Implementation Details:**
- **Embedding Model**: `Xenova/all-MiniLM-L6-v2` (23MB, browser-cached)
- **Vector Operations**: Mean pooling of theme embeddings
- **Similarity Metric**: Cosine similarity between pooled vectors
- **Fallback Strategy**: Exact text matching if embeddings fail

#### 5. Multi-Run Analysis Orchestration (`analyzeText`)
```typescript
const analyzeText = async (text: string, seed: number, runNumber: number) => {
  // Handles both single-chunk and multi-chunk analyses
  // Merges results intelligently across chunks
  // Provides accurate progress reporting
}
```

**Chunk Processing:**
- **Single Chunk**: Direct analysis path
- **Multi-Chunk**: Independent analysis + intelligent merging
- **Theme Deduplication**: Combines duplicate themes by name
- **Evidence Aggregation**: Collects unique evidence quotes
- **Sentiment Averaging**: Calculates mean sentiment scores

#### 6. Reliability Assessment (`synthesizeResults`)
```typescript
const synthesizeResults = async (allRuns: any[]) => {
  // Calculates inter-run similarities
  // Identifies consensus themes
  // Provides reliability metrics and warnings
}
```

**Consensus Logic:**
- **Threshold**: Themes must appear in ≥50% of runs
- **Adaptive**: Adjusts for partial data (fewer than 6 runs)
- **Ranking**: Sorts by consistency percentage
- **Evidence**: Collects supporting quotes from all occurrences

### Error Handling & Resilience

#### Retry Mechanism (`withRetry`)
```typescript
const withRetry = (fn: any, maxRetries: number = MAX_RETRIES) => {
  // Exponential backoff: 1.5s, 3s, 6s delays
  // Transparent error handling
  // Configurable retry attempts
}
```

**Retry Strategy:**
- **Max Retries**: 3 attempts per operation
- **Backoff Formula**: `1500ms * (2 ^ attempt)`
- **Error Categories**: Network, authentication, rate limiting
- **Graceful Failure**: Provides partial results when possible

#### Auto-Save System (`autoSaveRun`)
```typescript
const autoSaveRun = (runResult: any, runNumber: number) => {
  // Immediate download after each successful run
  // Prevents data loss on failures
  // Provides audit trail for analysis
}
```

**Data Persistence:**
- **Format**: JSON with metadata (timestamp, seed, themes, patterns)
- **Naming**: `run_{number}_{filename}_{date}.json`
- **Content**: Complete analysis results + file statistics
- **Recovery**: Enables manual review of individual runs

### State Management

#### React State Structure
```typescript
// Core Analysis State
const [apiKey, setApiKey] = useState('');
const [file, setFile] = useState(null);
const [status, setStatus] = useState('idle');
const [progress, setProgress] = useState(0);
const [currentRun, setCurrentRun] = useState(0);

// Results & Reliability
const [results, setResults] = useState(null);
const [partialResults, setPartialResults] = useState(null);
const [completedRuns, setCompletedRuns] = useState([]);

// Processing Statistics
const [fileStats, setFileStats] = useState(null);
const [cleaningStats, setCleaningStats] = useState<any>(null);

// Semantic Analysis
const extractorRef = useRef(null as any);
const [extractorReady, setExtractorReady] = useState(false);
```

#### Configuration Constants
```typescript
const MAX_RETRIES = 3;
const RETRY_DELAY_BASE = 1500;
const TOTAL_RUNS = 6;
const SEEDS = [42, 123, 456, 789, 1011, 1213];
```

### API Integration Details

#### Gemini API Endpoint
```
https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={API_KEY}
```

#### Request Structure
```json
{
  "contents": [{"parts": [{"text": "prompt"}]}],
  "generationConfig": {
    "temperature": 0.7,
    "topK": 40,
    "topP": 0.95,
    "maxOutputTokens": 8192
  }
}
```

#### Expected Response Format
```json
{
  "majorEmotionalThemes": [
    {
      "theme": "Theme Name",
      "description": "Detailed description",
      "sentiment": "positive|negative|neutral",
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
```

## 📊 Reliability Metrics

### Inter-Run Consistency Scoring
- **Method**: Semantic similarity using transformer embeddings
- **Range**: 0-1 (0% = no similarity, 100% = identical themes)
- **Interpretation**:
  - **High (>70%)**: Strong thematic convergence
  - **Moderate (50-70%)**: Reasonable consistency with variation
  - **Low (<50%)**: Considerable analytical divergence

### Consensus Theme Identification
- **Threshold**: Themes appearing in ≥50% of runs
- **Ranking**: By consistency percentage
- **Evidence**: Supporting quotes from all occurrences
- **Adaptive**: Adjusts for partial data sets

## 🔧 Setup & Installation

### Prerequisites
- Node.js 18+ 
- pnpm package manager
- Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd thematic-analysis-tool/something

# Install dependencies
pnpm install

# Development server
pnpm dev

# Production build
pnpm build
```

### Environment Configuration
```bash
# Optional: Create .env.local
GEMINI_API_KEY=your_api_key_here
```

## 🎨 UI/UX Features

### Design System
- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS with custom gradients
- **Icons**: Lucide React icon library
- **Responsive**: Mobile-first design approach

### User Interface Components
- **File Upload**: Drag-and-drop with validation
- **Progress Tracking**: Real-time progress bars and status
- **Results Display**: Interactive theme cards with evidence
- **Export Options**: Multiple format downloads
- **Error Handling**: User-friendly error messages with recovery options

## 📈 Performance Optimizations

### Browser Caching
- **Model Caching**: Hugging Face model cached after first download
- **API Response**: Intelligent caching of analysis results
- **File Processing**: Client-side text processing reduces server load

### Memory Management
- **Blob Cleanup**: Automatic URL.revokeObjectURL() for downloads
- **State Optimization**: Efficient React state updates
- **Chunk Processing**: Streaming analysis for large files

## 🔒 Security & Privacy

### Data Handling
- **Client-Side Processing**: All analysis performed in browser
- **No Server Storage**: Files never stored on external servers
- **API Key Security**: Stored in browser memory only
- **Local Downloads**: All exports saved locally

### API Security
- **HTTPS Only**: All API calls use secure connections
- **Rate Limiting**: Built-in retry logic handles API limits
- **Error Handling**: Secure error messages without data exposure

## 🚀 Deployment

### Build Configuration
```typescript
// next.config.ts
const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,  // Relaxed for rapid development
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
};
```

### TypeScript Configuration
```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": false,
    "suppressImplicitAnyIndexErrors": true,
    "noStrictGenericChecks": true
  }
}
```

## 📝 Usage Examples

### Basic Analysis Workflow
1. **Upload**: Drag and drop a .txt file
2. **Configure**: Enter Gemini API key
3. **Analyze**: Click "Start Analysis" 
4. **Monitor**: Watch real-time progress
5. **Review**: Examine consensus themes
6. **Export**: Download results in multiple formats

### Advanced Features
- **Resume**: Continue interrupted analyses
- **Partial Results**: Review completed runs during processing
- **Custom Seeds**: Modify SEEDS array for different variations
- **Chunk Size**: Adjust maxChunkSize for different file types

## 🔬 Research Applications

### Qualitative Research
- **Interview Analysis**: Process transcribed interviews
- **Focus Group Data**: Analyze group discussion transcripts
- **Survey Responses**: Extract themes from open-ended responses
- **Case Studies**: Identify patterns in narrative data

### Academic Use Cases
- **Literature Reviews**: Synthesize themes across papers
- **Content Analysis**: Analyze media or document collections
- **Ethnographic Research**: Process field notes and observations
- **Mixed Methods**: Complement quantitative findings

## 🤝 Contributing

### Development Guidelines
- **Code Style**: Follow existing patterns and conventions
- **Testing**: Test with various file sizes and formats
- **Documentation**: Update README for new features
- **Performance**: Monitor bundle size and runtime performance

### Feature Requests
- **Reliability Improvements**: Enhanced similarity algorithms
- **Export Formats**: Additional output formats (CSV, XML)
- **UI Enhancements**: Improved visualization and interaction
- **API Integration**: Support for additional language models

## 📄 License

This project is developed by Aza Lab for qualitative research applications. Please contact Aza Allsop and Nilesh Arnaiya for licensing and usage permissions.

## 🆘 Support

For technical support, feature requests, or research collaboration:
- **Contact**: Aza Allsop and Nilesh Arnaiya at Aza Lab at Yale university
- **Issues**: Report bugs and request features via GitHub issues
- **Documentation**: Refer to inline code comments for implementation details

---

*Built with ❤️ for the qualitative research community*