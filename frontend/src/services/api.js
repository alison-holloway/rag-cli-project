/**
 * API client for communicating with the RAG backend
 */

// Default URL for browser development
let API_BASE_URL = 'http://localhost:8000/api';

// Check if running in Tauri and get dynamic URL
const initApiUrl = async () => {
  if (window.__TAURI__) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const baseUrl = await invoke('get_api_url');
      API_BASE_URL = `${baseUrl}/api`;
      console.log('API URL set from Tauri:', API_BASE_URL);
    } catch (error) {
      console.warn('Failed to get API URL from Tauri, using default:', error);
    }
  }
};

// Initialize API URL
initApiUrl();

/**
 * Get the current API base URL
 * @returns {string} API base URL
 */
export function getApiBaseUrl() {
  return API_BASE_URL;
}

/**
 * Set the API base URL (used when backend starts on dynamic port)
 * @param {string} baseUrl - The new base URL (without /api suffix)
 */
export function setApiBaseUrl(baseUrl) {
  API_BASE_URL = `${baseUrl}/api`;
  console.log('API URL updated to:', API_BASE_URL);
}

/**
 * Query the knowledge base with a question
 * @param {string} query - The question to ask
 * @param {Object} options - Query options
 * @param {string} options.llmProvider - LLM provider ('ollama' or 'claude')
 * @param {number} options.topK - Number of chunks to retrieve
 * @param {number} options.temperature - LLM temperature
 * @param {boolean} options.showSources - Whether to include source information
 * @returns {Promise<Object>} Query response with answer, sources, and metadata
 */
export async function queryKnowledgeBase(query, options = {}) {
  const {
    llmProvider = 'ollama',
    topK = 5,
    temperature = 0.7,
    showSources = true,
  } = options;

  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      llm_provider: llmProvider,
      top_k: topK,
      temperature,
      show_sources: showSources,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Query failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Get health status of the RAG service
 * @returns {Promise<Object>} Health information
 */
export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  return response.json();
}

/**
 * List all indexed documents
 * @returns {Promise<Object>} Document list with statistics
 */
export async function listDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents`);

  if (!response.ok) {
    throw new Error(`Failed to list documents: ${response.status}`);
  }

  return response.json();
}

/**
 * Upload a document for indexing
 * @param {File} file - The file to upload
 * @param {boolean} force - Whether to re-index if document exists
 * @returns {Promise<Object>} Upload result
 */
export async function uploadDocument(file, force = false) {
  const formData = new FormData();
  formData.append('file', file);

  const url = new URL(`${API_BASE_URL}/upload`);
  if (force) {
    url.searchParams.set('force', 'true');
  }

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Delete a document from the knowledge base
 * @param {string} documentId - Document identifier (filename)
 * @returns {Promise<Object>} Deletion result
 */
export async function deleteDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Delete failed: ${response.status}`);
  }

  return response.json();
}
