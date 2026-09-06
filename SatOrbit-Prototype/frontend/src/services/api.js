const API_BASE_URL = 'http://localhost:8000';

/**
 * Fetches the list of active AOIs from the FastAPI backend.
 */
export async function fetchAOIs() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/aois`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to fetch AOIs: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('[API Service Error - fetchAOIs]:', error);
    throw error;
  }
}

/**
 * Sends a change detection request and prefixes returned image paths with the API host URL.
 */
export async function runChangeDetection(aoiId, date1, date2) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/change-detection`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ aoi_id: aoiId, date1, date2 }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Failed to execute change detection.');
    }

    const data = await response.json();

    // Helper to resolve relative path strings like '/api/outputs/...' to 'http://localhost:8000/api/outputs/...'
    const resolveUrl = (url) => (url && url.startsWith('/') ? `${API_BASE_URL}${url}` : url);

    return {
      ...data,
      raw_image_url: resolveUrl(data.raw_image_url),
      processed_image_url: resolveUrl(data.processed_image_url),
      clean_map_url: resolveUrl(data.clean_map_url),
      comparison_url: resolveUrl(data.comparison_url),
    };
  } catch (error) {
    console.error('[API Service Error - runChangeDetection]:', error);
    throw error;
  }
}