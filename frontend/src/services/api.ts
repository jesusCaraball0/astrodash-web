import axios, { AxiosError } from 'axios';

// Allow deployments to override the API host while keeping a sensible local default.
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export interface SpectrumData {
  x: number[];
  y: number[];
}

export interface ProcessParams {
  smoothing: number;
  knownZ: boolean;
  zValue?: number;
  minWave: number;
  maxWave: number;
  calculateRlap: boolean;
  file?: File;
  oscRef?: string;
  modelType?: 'dash' | 'transformer';
  model_id?: string;
}

export interface ProcessResponse {
  spectrum: {
    x: number[];
    y: number[];
    redshift?: number;
  };
  classification: {
    best_matches: Array<{
      type: string;
      age: string;
      probability: number;
      redshift?: number;
      rlap?: number | string;
      reliable: boolean;
    }>;
    best_match: {
      type: string;
      age: string;
      probability: number;
      redshift?: number;
    };
    reliable_matches: boolean;
  };
  model_type?: string;
}

export interface AnalysisOptionsResponse {
  sn_types: string[];
  age_bins?: string[];
  age_bins_by_type?: { [sn_type: string]: string[] };
}

export interface TemplateSpectrumResponse {
  x: number[];
  y: number[];
}

export interface LineListResponse {
  [element: string]: number[];
}

// Enhanced error interface for custom exceptions
export interface ApiError {
  detail: string;
  status_code?: number;
  message?: string;
}

// Error handling utility
const handleApiError = (error: AxiosError<ApiError>): never => {
  console.error('API Error:', {
    status: error.response?.status,
    statusText: error.response?.statusText,
    data: error.response?.data,
    message: error.message
  });

  // Extract error message from response
  let errorMessage = 'An unexpected error occurred.';

  if (error.response?.data?.detail) {
    errorMessage = error.response.data.detail;
  } else if (error.response?.data?.message) {
    errorMessage = error.response.data.message;
  } else if (error.message) {
    errorMessage = error.message;
  }

  // Create a custom error with the extracted message
  const customError = new Error(errorMessage);
  (customError as any).status = error.response?.status;
  (customError as any).response = error.response;

  throw customError;
};

class Api {
  async processSpectrum(params: ProcessParams): Promise<ProcessResponse> {
    console.log('API: Making request to /api/v1/process with params:', params);
    const formData = new FormData();

    // Add file if provided
    if (params.file) {
      console.log('API: Adding file to form data:', params.file.name);
      formData.append('file', params.file);
    }

    // Add other parameters
    const jsonParams: any = {
      smoothing: params.smoothing,
      knownZ: params.knownZ,
      zValue: params.zValue,
      minWave: params.minWave,
      maxWave: params.maxWave,
      calculateRlap: params.calculateRlap,
      oscRef: params.oscRef,
    };
    if (params.modelType) {
      jsonParams.modelType = params.modelType;
    } else {
      jsonParams.modelType = 'dash';
    }

    console.log('API: Adding JSON params to form data:', jsonParams);
    formData.append('params', JSON.stringify(jsonParams));

    // Add model_id as separate Form parameter if provided
    if (params.model_id) {
      formData.append('model_id', params.model_id);
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('API: Received response:', response.data);
      return response.data;
    } catch (error) {
      handleApiError(error as AxiosError<ApiError>);
      throw error; // This will never be reached but satisfies TypeScript
    }
  }

  async getAnalysisOptions(): Promise<AnalysisOptionsResponse> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/analysis-options`);
      return response.data;
    } catch (error) {
      handleApiError(error as AxiosError<ApiError>);
      throw error; // This will never be reached but satisfies TypeScript
    }
  }

  async getTemplateSpectrum(snType: string, age: string): Promise<TemplateSpectrumResponse> {
    console.log(`API: Fetching template for ${snType} ${age}`);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/template-spectrum`, {
        params: {
          sn_type: snType,
          age_bin: age
        }
      });
      console.log(`API: Template response received:`, response.data);
      return response.data;
    } catch (error: any) {
      console.error(`API: Template fetch failed for ${snType} ${age}:`, {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message
      });
      handleApiError(error as AxiosError<ApiError>);
      throw error; // This will never be reached but satisfies TypeScript
    }
  }

  async getLineList(): Promise<LineListResponse> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/line-list`);
      return response.data;
    } catch (error) {
      handleApiError(error as AxiosError<ApiError>);
      throw error; // This will never be reached but satisfies TypeScript
    }
  }

  async batchProcess({ zipFile, params }: { zipFile: File; params: any }): Promise<any> {
    const formData = new FormData();
    formData.append('zip_file', zipFile);

    // Ensure modelType is included in params
    const paramsWithModel = {
      ...params,
      modelType: params.modelType || 'dash'  // Default to dash if not specified
    };

    formData.append('params', JSON.stringify(paramsWithModel));

    // Add model_id as separate Form parameter if provided
    if (params.model_id) {
      formData.append('model_id', params.model_id);
    }
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/batch-process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      handleApiError(error as AxiosError<ApiError>);
      throw error; // This will never be reached but satisfies TypeScript
    }
  }

  async batchProcessMultiple({ files, params }: { files: File[]; params: any }): Promise<any> {
    const formData = new FormData();

    // Add each file to the form data
    files.forEach((file, index) => {
      formData.append(`files`, file);
    });

    // Ensure modelType is included in params
    const paramsWithModel = {
      ...params,
      modelType: params.modelType || 'dash'  // Default to dash if not specified
    };

    formData.append('params', JSON.stringify(paramsWithModel));

    // Add model_id as separate Form parameter if provided
    if (params.model_id) {
      formData.append('model_id', params.model_id);
    }
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/batch-process-multiple`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      handleApiError(error as AxiosError<ApiError>);
      throw error; // This will never be reached but satisfies TypeScript
    }
  }

  async uploadModel({ file, classMapping, inputShape, name, description }: {
    file: File,
    classMapping: object,
    inputShape: number[],
    name?: string,
    description?: string
  }) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('class_mapping', JSON.stringify(classMapping));
    formData.append('input_shape', JSON.stringify(inputShape));
    if (name) formData.append('name', name);
    if (description) formData.append('description', description);
    const response = await axios.post(`${API_BASE_URL}/api/v1/upload-model`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getUserModels(): Promise<any[]> {
    // Fetch user-uploaded models from the backend
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/models`);
      return response.data.models || response.data || [];
    } catch (error) {
      console.warn('Failed to fetch user models:', error);
      // Return empty array if models endpoint fails
      return [];
    }
  }

  async deleteModel(modelId: string): Promise<any> {
    try {
      const response = await axios.delete(`${API_BASE_URL}/api/v1/models/${modelId}`);
      return response.data;
    } catch (error) {
      console.error('API: Error deleting model:', error);
      throw error;
    }
  }

  async updateModel(modelId: string, updates: { name?: string; description?: string }): Promise<any> {
    try {
      const formData = new FormData();
      if (updates.name) formData.append('name', updates.name);
      if (updates.description) formData.append('description', updates.description);

      const response = await axios.put(`${API_BASE_URL}/api/v1/models/${modelId}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    } catch (error) {
      console.error('API: Error updating model:', error);
      throw error;
    }
  }
}

export const api = new Api();
